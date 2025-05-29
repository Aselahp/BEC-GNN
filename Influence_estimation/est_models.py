
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import GINConv, GATConv, SAGEConv, GCNConv

DEVICE = "cpu"

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
    def __init__(self, hidden_dim=64):
        super(WeightMLP, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.layers = None
        
    def build_network(self, input_dim):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        input_dim = x.shape[1] if len(x.shape) > 1 else x.shape[0]
        
        if self.layers is None or self.layers[0].in_features != input_dim:
            self.build_network(input_dim)
            
        if len(original_shape) == 1:
            x = x.reshape(1, -1)
        output = self.layers(x)
        
        if len(original_shape) == 1:
            output = output.squeeze(0)
            
        return output

class CurvatureMLP(nn.Module):
    def __init__(self, nfeat, nhid):
        super(CurvatureMLP, self).__init__()
        self.curv_mlp = MLP(nfeat, nhid, 1)

    def forward(self, x):
        output = self.curv_mlp(x)
        return torch.sigmoid(output)

def compute_gamma(f, edge_index, weights):
    src, dst = edge_index[0], edge_index[1]    
    f_diff_squared = (f[dst] - f[src])**2
    weighted_diffs = weights * f_diff_squared
    result = torch.zeros_like(f)
    result.scatter_add_(0, src, weighted_diffs)
    return 0.5 * result


def compute_gamma2_optimized(f, edge_index, weights):
    n_vertices = f.size(0)
    src, dst = edge_index[0], edge_index[1]
    f_diff = f[dst] - f[src]
    
    delta_f = torch.zeros_like(f)
    weighted_diffs = weights * f_diff
    delta_f.scatter_add_(0, src, weighted_diffs)
    
    gamma_f = torch.zeros_like(f)
    weighted_squared_diffs = weights * f_diff.pow(2)
    gamma_f.scatter_add_(0, src, weighted_squared_diffs)
    gamma_f = 0.5 * gamma_f
    
    gamma_diff = gamma_f[dst] - gamma_f[src]
    delta_gamma = torch.zeros_like(f)
    weighted_gamma_diffs = weights * gamma_diff
    delta_gamma.scatter_add_(0, src, weighted_gamma_diffs)
    
    delta_f_diff = delta_f[dst] - delta_f[src]
    gamma_f_delta = torch.zeros_like(f)
    weighted_cross_diffs = weights * f_diff * delta_f_diff
    gamma_f_delta.scatter_add_(0, src, weighted_cross_diffs)
    gamma_f_delta = 0.5 * gamma_f_delta
    gamma2 = 0.5*delta_gamma - gamma_f_delta
    
    return gamma2


def compute_loss(kappa, gamma, gamma2):
    kappa_gamma = kappa * gamma
    diff = kappa_gamma - gamma2
    loss_per_node = torch.relu(diff)
    total_loss = loss_per_node.sum() - kappa.sum()   
    return total_loss


    
class DC_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, adj, fn_count = 3):
        super(DC_GCN, self).__init__()
        
        self.nn  = nn.ModuleList([
            GCNConv(in_channels if i == 0 else hidden_channels, out_channels if i == num_layers - 1 else hidden_channels) 
            for i in range(num_layers)
            ])
        self.dropout = dropout
        self.num_layers = num_layers
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(in_channels, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(in_channels, 20))
 
        
    
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = int(n_vertices * k / 100)
        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)
        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0
        src, dst = edge_index[0], edge_index[1]
        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        modified_edge_index = edge_index[:, ~removed_edge_mask]
        return modified_edge_index

    def forward(self, x, edge_index, p=10):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        weights = self.weight_model(torch.ones(edge_index.size(1)).float())
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, edge_index, weights)
            gamma2_f = compute_gamma2_optimized(f, edge_index, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            

    
        for i in range(self.num_layers-1):
            edge_index = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p * (i))
            x = self.nn[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
        edge_index = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p * (i+1))
        x =self.nn[-1](x, edge_index)
        x = torch.sigmoid(x).mean(dim=1, keepdim=True)
        return x, curv_loss
    
    
class DC_GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, adj, fn_count = 3):
        super(DC_GraphSAGE, self).__init__()
        
        self.nn  = nn.ModuleList([
            SAGEConv(in_channels if i == 0 else hidden_channels, out_channels if i == num_layers - 1 else hidden_channels) 
            for i in range(num_layers)
            ])
        self.dropout = dropout
        self.num_layers = num_layers
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(in_channels, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(in_channels, 20))
 
        
    
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = int(n_vertices * k / 100)

        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)

        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0

        src, dst = edge_index[0], edge_index[1]

        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        modified_edge_index = edge_index[:, ~removed_edge_mask]

        return modified_edge_index

    def forward(self, x, edge_index, p=10):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        weights = self.weight_model(torch.ones(edge_index.size(1)).float())
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, edge_index, weights)
            gamma2_f = compute_gamma2_optimized(f, edge_index, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            

    
        for i in range(self.num_layers-1):
            edge_index = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p * (i))
            x = self.nn[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
        edge_index = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p * (i+1))
        x = self.nn[-1](x, edge_index)
        x = torch.sigmoid(x).mean(dim=1, keepdim=True)
        return x, curv_loss
    
    
class DC_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, adj, fn_count = 3):
        super(DC_GAT, self).__init__()
        
        self.nn  = nn.ModuleList([
            GATConv(in_channels if i == 0 else hidden_channels, out_channels if i == num_layers - 1 else hidden_channels) 
            for i in range(num_layers)
            ])
        self.dropout = dropout
        self.num_layers = num_layers
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(in_channels, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(in_channels, 20))
 
        
    
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = int(n_vertices * k / 100)

        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)

        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0

        src, dst = edge_index[0], edge_index[1]

        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)

        modified_edge_index = edge_index[:, ~removed_edge_mask]

        return modified_edge_index

    def forward(self, x, edge_index, p=5):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        weights = self.weight_model(torch.ones(edge_index.size(1)).float())
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, edge_index, weights)
            gamma2_f = compute_gamma2_optimized(f, edge_index, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            

    
        for i in range(self.num_layers-1):
            edge_index = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p * (i))
            x = F.relu(self.nn[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
    
        edge_index = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p * (i+1))
        x = self.nn[-1](x, edge_index)
        x = torch.sigmoid(x).mean(dim=1, keepdim=True)
        return x, curv_loss