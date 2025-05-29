import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from layers import GIN, GCN, GraphSN, NCGNN
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class WeightMLP(nn.Module):
    def __init__(self, hidden_dims=[64, 32]):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(1, hidden_dims[0]),
            nn.ReLU()
        ])
        
        for i in range(len(hidden_dims) - 1):
            self.layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU()
            ])

        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        
    def forward(self, adj_matrix):
        batch_size, n_vertices, _ = adj_matrix.shape
        
        x = adj_matrix.view(-1, 1)  

        for layer in self.layers:
            x = layer(x)

        x = torch.sigmoid(x)

        x = x.view(batch_size, n_vertices, n_vertices)
        
        return x

    
class CurvatureMLP(nn.Module):
    def __init__(self, nfeat, nhid):
        super(CurvatureMLP, self).__init__()
        self.curv_mlp = MLP(nfeat, nhid, 1)

    def forward(self, x):
        output = self.curv_mlp(x)
        return torch.sigmoid(output) 


def compute_gamma(f, adj_matrix, weights):
    f_expanded_1 = f.unsqueeze(-1)
    f_expanded_2 = f.unsqueeze(1)
    f_diff_squared = (f_expanded_2 - f_expanded_1)**2
    weighted_adj = adj_matrix * weights
    return 0.5 * torch.sum(weighted_adj * f_diff_squared, dim=2)


def compute_gamma2_optimized(f, adj_matrix, weights):
    n_graphs, n_vertices = f.shape
    weighted_adj = adj_matrix * weights    
    f_expanded = f.unsqueeze(2).expand(-1, -1, n_vertices)
    f_diff = f_expanded - f_expanded.transpose(1, 2)
    
    delta_f = torch.sum(weighted_adj * f_diff, dim=2)
    gamma_f = 0.5 * torch.sum(weighted_adj * f_diff.pow(2), dim=2)

    gamma_y = gamma_f.unsqueeze(2).expand(-1, -1, n_vertices)
    delta_gamma = torch.sum(weighted_adj * (gamma_y - gamma_f.unsqueeze(1)), dim=2)

    delta_f_expanded = delta_f.unsqueeze(2).expand(-1, -1, n_vertices)
    delta_f_diff = delta_f_expanded - delta_f_expanded.transpose(1, 2)
    gamma_f_delta = 0.5 * torch.sum(weighted_adj * f_diff * delta_f_diff, dim=2)
    gamma2 = 0.5*delta_gamma - gamma_f_delta
    
    return gamma2

def compute_loss(kappa, gamma, gamma2):
    kappa_gamma = kappa.squeeze(-1) * gamma  
    diff = kappa_gamma - gamma2  
    loss_per_node = torch.relu(diff)  
    loss_per_graph = loss_per_node.sum(dim=1)
    total_loss = loss_per_graph.sum() - kappa.sum()  
    return total_loss

class GNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batch_size, batchnorm_dim, dropout_1, dropout_2, fn_count = 3):
        super().__init__()
        
        self.dropout = dropout_1
        
        self.convs = nn.ModuleList()
        
        self.convs.append(GIN(input_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        for _ in range(n_layers-1):
            self.convs.append(GIN(hidden_dim, hidden_dim, batchnorm_dim, dropout_2))
        self.out_proj = nn.Linear((input_dim+hidden_dim*(n_layers)), output_dim)
        
        self.curv_mlp = CurvatureMLP(input_dim, hidden_dim)
        self.weight_mlp = WeightMLP(hidden_dims=[64, 32])
        #self.weights = torch.nn.Parameter(torch.randn_like(adj))
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(input_dim, hidden_dim))
            
    def remove_top_k_neighbors_with_mask(self, adj_matrix, curvatures, k):
        n_graphs, n_vertices = adj_matrix.shape[:2]
        num_to_remove = int(n_vertices * k / 100)
        if curvatures.dim() == 3:
            curvatures = curvatures.squeeze(-1)

        _, top_indices = torch.topk(curvatures, num_to_remove, dim=1)
 
        mask = torch.ones((n_graphs, n_vertices), device=adj_matrix.device)

        batch_indices = torch.arange(n_graphs).unsqueeze(1).expand(-1, num_to_remove)
        mask[batch_indices, top_indices] = 0

        mask_row = mask.unsqueeze(2)
        mask_col = mask.unsqueeze(1)
        sampled_adj = adj_matrix * mask_row * mask_col  
        return sampled_adj          

    def forward(self, data, p=10):
        X, A = data[:2]

        hidden_states = [X]
        
        curv_loss = 0
        kappa = self.curv_mlp(X)
        weights = self.weight_mlp(A) * A
        for i in self.fn_mlp:
            f = i(X).squeeze(-1)
            gamma_f = compute_gamma(f, A, weights)
            gamma2_f = compute_gamma2_optimized(f, A, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
        
        t = 1
        for layer in self.convs:
            X = F.dropout(layer(A, X), self.dropout)
            A = self.remove_top_k_neighbors_with_mask(A, kappa, p * (t))
            hidden_states.append(X)
            t = t+1

        X = torch.cat(hidden_states, dim=2).sum(dim=1)
        X = self.out_proj(X)

        return X, curv_loss