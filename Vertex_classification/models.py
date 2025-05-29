import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, SGConv, MixHopConv, DirGNNConv

DEVICE = "cpu"



def compute_gamma(f, adj_matrix, weights):
    f_diff_squared = (f.unsqueeze(0) - f.unsqueeze(1))**2
    weighted_adj = adj_matrix * weights
    return 0.5 * torch.sum(weighted_adj * f_diff_squared, dim=1)



def compute_gamma2_optimized(f, adj_matrix, weights):

    weighted_adj = adj_matrix * weights
    n_vertices = f.shape[0]
    f_expanded = f.unsqueeze(0).expand(n_vertices, -1)  
    f_diff = f_expanded - f_expanded.t() 
    delta_f = torch.sum(weighted_adj * f_diff, dim=1)
    gamma_f = 0.5 * torch.sum(weighted_adj * f_diff.pow(2), dim=1) 
    gamma_y = gamma_f.unsqueeze(0).expand(n_vertices, -1) 
    delta_gamma = torch.sum(weighted_adj * (gamma_y - gamma_f.unsqueeze(1)), dim=1)
    delta_f_expanded = delta_f.unsqueeze(0).expand(n_vertices, -1)  
    delta_f_diff = delta_f_expanded - delta_f_expanded.t()  
    gamma_f_delta = 0.5 * torch.sum(weighted_adj * f_diff * delta_f_diff, dim=1) 
    gamma2 = 0.5*delta_gamma - gamma_f_delta    
    return gamma2


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class CurvatureMLP(nn.Module):
    def __init__(self, nfeat, nhid):
        super(CurvatureMLP, self).__init__()
        self.curv_mlp = MLP(nfeat, nhid, 1)

    def forward(self, x):
        output = self.curv_mlp(x)
        return torch.sigmoid(output)


def compute_loss(kappa, gamma, gamma2):
    kappa_gamma = kappa * gamma 
    diff = kappa_gamma - gamma2 
    loss_per_node = torch.relu(diff)
    total_loss = loss_per_node.sum()
    return total_loss
         


class DC_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, iterations, adj, fn_count = 1):
        super(DC_GCN, self).__init__()
        
        self.nn = self.nn = nn.ModuleList([
            GCNConv(nfeat if i == 0 else nhid, nclass if i == iterations - 1 else nhid) 
            for i in range(iterations)
            ])
        self.dropout = dropout
        self.curv_mlp = CurvatureMLP(nfeat, nhid)
        self.iterations = iterations
        self.fn_mlp = []
        self.weights = torch.nn.Parameter(torch.randn_like(adj))
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(nfeat, nhid))
 
    
    def remove_top_k_neighbors_with_mask(self, adj_matrix, curvatures, k):
        n = adj_matrix.shape[0]
        num_to_remove = int(n * k / 100)
        curvatures = curvatures.squeeze(-1)

        _, top_indices = torch.topk(curvatures, num_to_remove)

        mask = torch.ones_like(curvatures)
        mask[top_indices] = 0  
    
        mask_row = mask.unsqueeze(1)  
        mask_col = mask.unsqueeze(0)  

        sampled_adj = adj_matrix * mask_row * mask_col

        return sampled_adj

    def forward(self, x, adj, p=40):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        edge_weights = torch.sigmoid(self.weights)
        for i in self.fn_mlp:
            f = i(x).squeeze(-1)
            gamma_f = compute_gamma(f, adj, edge_weights)
            gamma2_f = compute_gamma2_optimized(f, adj, edge_weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
        
        
        for t in range(self.iterations):
            sampled_adj = self.remove_top_k_neighbors_with_mask(adj, kappa, p * (t))
            x = F.relu(self.nn[t](x, sampled_adj.nonzero(as_tuple=False).t()))
            if t < (self.iterations-1):
                x = F.dropout(x, self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1), 1 * curv_loss
    
class DC_GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, iterations, adj, fn_count = 2):
        super(DC_GraphSAGE, self).__init__()
        
        self.nn = self.nn = nn.ModuleList([
            SAGEConv(nfeat if i == 0 else nhid, nclass if i == iterations - 1 else nhid) 
            for i in range(iterations)
            ])
        self.dropout = dropout
        self.curv_mlp = CurvatureMLP(nfeat, nhid)
        self.weights = torch.nn.Parameter(torch.randn_like(adj))
        self.iterations = iterations
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(nfeat, nhid))
 
    
    def remove_top_k_neighbors_with_mask(self, adj_matrix, curvatures, k):
        n = adj_matrix.shape[0]
        num_to_remove = int(n * k / 100)
        curvatures = curvatures.squeeze(-1)

        _, top_indices = torch.topk(curvatures, num_to_remove)

        mask = torch.ones_like(curvatures)
        mask[top_indices] = 0
        mask_row = mask.unsqueeze(1) 
        mask_col = mask.unsqueeze(0)
        sampled_adj = adj_matrix * mask_row * mask_col

        return sampled_adj

    def forward(self, x, adj, p=5):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        edge_weights = torch.sigmoid(self.weights)
        for i in self.fn_mlp:
            f = i(x).squeeze(-1)
            gamma_f = compute_gamma(f, adj, edge_weights)
            gamma2_f = compute_gamma2_optimized(f, adj, edge_weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
        
        for t in range(self.iterations):
            sampled_adj = self.remove_top_k_neighbors_with_mask(adj, kappa, p * (t))
            x = self.nn[t](x, sampled_adj.nonzero(as_tuple=False).t())
            if t < (self.iterations-1):
                x = F.dropout(x, self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1), 1 * curv_loss
    
    
    
class DC_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, iterations, adj, fn_count = 3):
        super(DC_GAT, self).__init__()
        
        self.nn = self.nn = nn.ModuleList([
            GATConv(nfeat if i == 0 else nhid, nclass if i == iterations - 1 else nhid) 
            for i in range(iterations)
            ])
        self.dropout = dropout
        self.curv_mlp = CurvatureMLP(nfeat, nhid)
        self.weights = torch.nn.Parameter(torch.randn_like(adj))
        self.iterations = iterations
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(nfeat, nhid))
 
        
    
    def remove_top_k_neighbors_with_mask(self, adj_matrix, curvatures, k):
        n = adj_matrix.shape[0]
        num_to_remove = int(n * k / 100)
        curvatures = curvatures.squeeze(-1)

        _, top_indices = torch.topk(curvatures, num_to_remove)
        mask = torch.ones_like(curvatures)
        mask[top_indices] = 0  
        mask_row = mask.unsqueeze(1) 
        mask_col = mask.unsqueeze(0)  
        sampled_adj = adj_matrix * mask_row * mask_col

        return sampled_adj

    def forward(self, x, adj, p=5):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        edge_weights = torch.sigmoid(self.weights)
        for i in self.fn_mlp:
            f = i(x).squeeze(-1)
            gamma_f = compute_gamma(f, adj, edge_weights)
            gamma2_f = compute_gamma2_optimized(f, adj, edge_weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
        
        
        for t in range(self.iterations):
            sampled_adj = self.remove_top_k_neighbors_with_mask(adj, kappa, p * (t))
            x = self.nn[t](x, sampled_adj.nonzero(as_tuple=False).t())
            if t < (self.iterations-1):
                x = F.dropout(x, self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1), 1 * curv_loss
    
    
class DC_SGC(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, iterations, adj, fn_count=3):
        super(DC_SGC, self).__init__()
        
        self.fc1 = nn.Linear(nfeat, nhid)
        self.nn = nn.ModuleList([
            SGConv(nfeat if i == 0 else nhid, nclass if i == iterations - 1 else nhid)
            for i in range(iterations)
        ])
        self.dropout = dropout
        self.iterations = iterations
        self.curv_mlp = CurvatureMLP(nfeat, nhid)
        self.weights = torch.nn.Parameter(torch.randn_like(adj))
        self.fn_mlp = nn.ModuleList([CurvatureMLP(nfeat, nhid) for _ in range(fn_count)])

    def remove_top_k_neighbors_with_mask(self, adj_matrix, curvatures, k):
        n = adj_matrix.shape[0]
        num_to_remove = int(n * k / 100)
        curvatures = curvatures.squeeze(-1)

        _, top_indices = torch.topk(curvatures, num_to_remove)
        mask = torch.ones_like(curvatures)
        mask[top_indices] = 0 
        mask_row = mask.unsqueeze(1) 
        mask_col = mask.unsqueeze(0) 
        sampled_adj = adj_matrix * mask_row * mask_col
        return sampled_adj

    def forward(self, x, adj, p=5):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        edge_weights = torch.sigmoid(self.weights)
        for i in self.fn_mlp:
            f = i(x).squeeze(-1)
            gamma_f = compute_gamma(f, adj, edge_weights)
            gamma2_f = compute_gamma2_optimized(f, adj, edge_weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)

        for t in range(self.iterations):
            sampled_adj = self.remove_top_k_neighbors_with_mask(adj, kappa, p * (t))
            x = self.nn[t](x, sampled_adj.nonzero(as_tuple=False).t())
            if t < (self.iterations - 1):
                x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=-1), 1 * curv_loss

class DC_MixHop(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, iterations, adj, fn_count = 3):
        super(DC_MixHop, self).__init__()
        
        self.nn = self.nn = nn.ModuleList([
            MixHopConv(nfeat if i == 0 else nhid, nclass if i == iterations - 1 else nhid) 
            for i in range(iterations)
            ])
        self.dropout = dropout
        self.curv_mlp = CurvatureMLP(nfeat, nhid)
        self.iterations = iterations
        self.fn_mlp = []
        self.weights = torch.nn.Parameter(torch.randn_like(adj))
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(nfeat, nhid))
 
    
    def remove_top_k_neighbors_with_mask(self, adj_matrix, curvatures, k):
        n = adj_matrix.shape[0]
        num_to_remove = int(n * k / 100)
        curvatures = curvatures.squeeze(-1)

        _, top_indices = torch.topk(curvatures, num_to_remove)

        mask = torch.ones_like(curvatures)
        mask[top_indices] = 0  
    
        mask_row = mask.unsqueeze(1)  
        mask_col = mask.unsqueeze(0)  

        sampled_adj = adj_matrix * mask_row * mask_col

        return sampled_adj

    def forward(self, x, adj, p=30):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        edge_weights = torch.sigmoid(self.weights)
        for i in self.fn_mlp:
            f = i(x).squeeze(-1)
            gamma_f = compute_gamma(f, adj, edge_weights)
            gamma2_f = compute_gamma2_optimized(f, adj, edge_weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
        
        
        for t in range(self.iterations):
            sampled_adj = self.remove_top_k_neighbors_with_mask(adj, kappa, p * (t))
            x = F.relu(self.nn[t](x, sampled_adj.nonzero(as_tuple=False).t()))
            if t < (self.iterations-1):
                x = F.dropout(x, self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1), 1 * curv_loss

class DC_DirGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, iterations, adj, fn_count = 3):
        super(DC_DirGNN, self).__init__()
        
        self.nn = self.nn = nn.ModuleList([
            DirGNNConv(GCNConv(nfeat if i == 0 else nhid, nclass if i == iterations - 1 else nhid))
            for i in range(iterations)
            ])
        self.dropout = dropout
        self.curv_mlp = CurvatureMLP(nfeat, nhid)
        self.iterations = iterations
        self.fn_mlp = []
        self.weights = torch.nn.Parameter(torch.randn_like(adj))
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(nfeat, nhid))
 
    
    def remove_top_k_neighbors_with_mask(self, adj_matrix, curvatures, k):
        n = adj_matrix.shape[0]
        num_to_remove = int(n * k / 100)
        curvatures = curvatures.squeeze(-1)

        _, top_indices = torch.topk(curvatures, num_to_remove)

        mask = torch.ones_like(curvatures)
        mask[top_indices] = 0  
    
        mask_row = mask.unsqueeze(1)  
        mask_col = mask.unsqueeze(0)  

        sampled_adj = adj_matrix * mask_row * mask_col

        return sampled_adj

    def forward(self, x, adj, p=30):
        curv_loss = 0
        kappa = self.curv_mlp(x)
        edge_weights = torch.sigmoid(self.weights)
        for i in self.fn_mlp:
            f = i(x).squeeze(-1)
            gamma_f = compute_gamma(f, adj, edge_weights)
            gamma2_f = compute_gamma2_optimized(f, adj, edge_weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
        
        
        for t in range(self.iterations):
            sampled_adj = self.remove_top_k_neighbors_with_mask(adj, kappa, p * (t))
            x = F.relu(self.nn[t](x, sampled_adj.nonzero(as_tuple=False).t()))
            if t < (self.iterations-1):
                x = F.dropout(x, self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1), 1 * curv_loss
    