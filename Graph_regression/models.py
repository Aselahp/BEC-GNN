import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from layers import *
from torch.nn import Linear, Sequential, ReLU, Sigmoid, Dropout, BatchNorm1d

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


class GNN_bench(nn.Module):  # for zinc dataset
    
    def __init__(self, params):
        super().__init__()
        
        self.nfeat = params['nfeat']
        self.nhid= params['nhid']
        self.nlayers = params['nlayers']
        self.nclass = params['nclass']

        self.readout = params['readout']
        self.dropout = params['dropout']
        self.jk = params['jk']
        
        fn_count = 3 # set function count

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(1, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(1, 20))

        self.embedding_h = nn.Linear(self.nfeat, self.nhid)
        
        for layer in range(self.nlayers):
            self.convs.append(GINConv(self.nhid))
            self.batch_norms.append(BatchNorm1d(self.nhid))
            
            
        # pooler
        if self.readout == "sum":
            self.pool = global_add_pool
        elif self.readout == "mean":
            self.pool = global_mean_pool
        else:
            raise NotImplementedError

        if self.nclass > 1:
            if self.jk:
                self.linears_prediction = torch.nn.ModuleList()
                for layer in range(self.nlayers+1):
                    self.linears_prediction.append(nn.Linear(self.nhid*5, self.nclass))
            else:
                self.linears_prediction = nn.Linear(self.nhid*5, self.nclass)
        else: # mlp readout for zinc
            hidden_multiplier = params['multiplier']
            if self.jk:
                self.linears_prediction = torch.nn.ModuleList()
                for layer in range(self.nlayers+1):
                    self.linears_prediction.append(nn.Linear(self.nhid*(self.nlayers), hidden_multiplier * self.nhid))
            else:
                self.linears_prediction = nn.Linear(self.nhid, hidden_multiplier * self.nhid)
            self.fc2 = nn.Linear(hidden_multiplier * self.nhid, self.nclass)
            
    def remove_top_k_neighbors_with_mask(self, edge_index, edge_weight, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = int(n_vertices * k / 100)
        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)
        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0
        src, dst = edge_index[0], edge_index[1]
        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        modified_edge_index = edge_index[:, ~removed_edge_mask]
        modified_edge_weight = edge_weight[~removed_edge_mask]

        return modified_edge_index, modified_edge_weight
        

    def forward(self, h, batched_data, batch_idx, p=10):
        h = self.embedding_h(h)
        batched_data = batched_data
        #x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        #x, edge_index, edge_attr, norm_edge_weight, norm_self_loop, batch = batched_data.x, batched_data.edge_index,  batched_data.edge_attr, batched_data.norm_edge_weight, batched_data.norm_self_loop, batched_data.batch
        x, edge_index, edge_attr, norm_edge_weight, batch = batched_data.x, batched_data.edge_index,  batched_data.edge_attr, batched_data.norm_edge_weight, batched_data.batch
        
        
        curv_loss = 0
        kappa = self.curv_mlp(x.float())
        weights = self.weight_model(edge_attr.float())
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, edge_index, weights)
            gamma2_f = compute_gamma2_optimized(f, edge_index, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
        
        h_list = [h]
        
        for layer in range(len(self.convs)):
            #edge_index = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p *(layer))
            edge_index, norm_edge_weight= self.remove_top_k_neighbors_with_mask(edge_index, norm_edge_weight, kappa, p *(layer))
            h = self.convs[layer](h, edge_index)
            #h = self.convs[layer](h, edge_index, norm_edge_weight, norm_self_loop)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h = h + h_list[layer]  # residual 
            h = F.dropout(h, self.dropout, training=self.training)
            h_list.append(h)
            
            

        X = self.pool(h_list[-1], batch_idx)
            

        if self.jk:  # waste of parameter budget for zinc
            h = 0
            for layer in range(self.nlayers + 1):
                h += self.linears_prediction[layer](self.pool(h_list[layer], batch_idx))
        else:
            h = self.linears_prediction(self.pool(h_list[-1], batch_idx))
            
        
        h = F.relu(h)
        h = self.fc2(h)
        return h, curv_loss
