import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from collections import defaultdict

import math

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
        # MLP definition
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

class GraphSN(MessagePassing):
    def __init__(self, emb_dim):
        super(GraphSN, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), 
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        
        self.eps = torch.nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, x, edge_index, edge_attr, norm_edge_weight, norm_self_loop, modified_edge_weights):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(self.eps * norm_self_loop.view(-1,1) * x + self.propagate(edge_index, x=x, 
                                                                                 edge_attr=edge_embedding, norm=norm_edge_weight, norm2=modified_edge_weights))
        return out

    def message(self, x_j, edge_attr, norm, norm2):
        return norm2.view(-1, 1) * norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weights):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=edge_weights))

        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, residual = False, gnn_type = 'graphSN', fn_count = 1):
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        ### add residual connection or not
        self.residual = residual
        self.gnn_type = gnn_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(emb_dim, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(emb_dim, 20))

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        if gnn_type == 'gin':
            self.fc = nn.Linear(3, emb_dim)

        for layer in range(num_layer):
            if gnn_type == 'graphSN':
                self.convs.append(GraphSN(emb_dim))
            elif gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = min(int(n_vertices * k / 100), 100)
        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)
        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0
        src, dst = edge_index[0], edge_index[1]
        edge_weights = torch.ones(edge_index.size(1), device=edge_index.device)
        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        edge_weights[removed_edge_mask] = 0.00001 #make edge weights to near zero
    
        return edge_weights

    def forward(self, batched_data, p=10):
        if self.gnn_type == 'graphSN':
            x, edge_index, edge_attr, norm_edge_weight, norm_self_loop, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.norm_edge_weight, batched_data.norm_self_loop, batched_data.batch
        else:
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x = self.atom_encoder(x)
        if self.gnn_type == 'gin':
            x = x + self.fc(batched_data.c.float())
        curv_loss = 0
        kappa = self.curv_mlp(x.float())
        weights = self.weight_model(edge_attr.sum(dim=1).float())
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, edge_index, weights)
            gamma2_f = compute_gamma2_optimized(f, edge_index, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            
        ### computing input node embedding

        h_list = [x]
        for layer in range(self.num_layer):
            modified_edge_weights = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p *(layer))
            if self.gnn_type == 'graphSN':
                h = self.convs[layer](h_list[layer], edge_index, edge_attr, norm_edge_weight, norm_self_loop, modified_edge_weights)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr, modified_edge_weights)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        node_representation = 0
        for layer in range(self.num_layer + 1):
            node_representation += h_list[layer]

        return node_representation, curv_loss


if __name__ == "__main__":
    pass