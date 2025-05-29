import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
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
        self.curv_mlp = MLP(nfeat, nhid, 1)

    def forward(self, x):
        output = self.curv_mlp(x)
        return torch.sigmoid(output) 

def compute_laplacian(f, adj_matrix):
    weighted_adj = adj_matrix    
    f_diff = f.unsqueeze(0) - f.unsqueeze(1)
    return torch.sum(weighted_adj * f_diff, dim=1)

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


def normalize_adjacency(adj):
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    degree = torch.sum(adj, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  #
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    normalized_adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    return normalized_adj

def compute_loss(kappa, gamma, gamma2):
    kappa_gamma = kappa * gamma

    diff = kappa_gamma - gamma2

    loss_per_node = torch.relu(diff)

    total_loss = loss_per_node.sum() - kappa.sum()
    
    return total_loss


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
    

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weights):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm, edge_weights=edge_weights) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm, edge_weights):
        return edge_weights.view(-1, 1) * norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    
class NCGNNConv(MessagePassing):
    def __init__(self, emb_dim, batch_size=1024):
        super(NCGNNConv, self).__init__(aggr="add")
        
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim),
            torch.nn.BatchNorm1d(2*emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*emb_dim, emb_dim)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim),
            torch.nn.LayerNorm(2*emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*emb_dim, emb_dim)
        )
        
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.batch_size = batch_size
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weights):
        edge_embedding = self.bond_encoder(edge_attr)
        neighbor_sum = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=edge_weights)
        
        mask = edge_weights >= 1
        pruned_edge_index = edge_index[:, mask]
        pair_interactions = self.compute_pair_interactions(x, pruned_edge_index)
        out = self.mlp1((1 + self.eps) * x + neighbor_sum + pair_interactions)
        return out

    def compute_pair_interactions(self, x, edge_index):
        num_nodes = x.size(0)
        device = x.device
        
        edge_set = {(row.item(), col.item()) for row, col in zip(*edge_index)}
        edge_set.update({(col.item(), row.item()) for row, col in zip(*edge_index)})
        
        neighbors = defaultdict(list)
        for row, col in zip(*edge_index):
            neighbors[row.item()].append(col.item())
            
        pair_interactions = torch.zeros_like(x)
        for start_idx in range(0, num_nodes, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_nodes)
            batch_pairs = []
            batch_nodes = []
            batch_features = []
            
            for node_idx in range(start_idx, end_idx):
                node_neighbors = neighbors[node_idx]
                for i, u1 in enumerate(node_neighbors):
                    for u2 in node_neighbors[i+1:]:
                        if (u1, u2) in edge_set:
                            batch_pairs.append((node_idx, u1, u2))
                            batch_nodes.append(node_idx)
                            batch_features.append(x[u1] + x[u2])
            
            if not batch_pairs:
                continue
                
            batch_nodes = torch.tensor(batch_nodes, device=device)
            batch_features = torch.stack(batch_features)
            processed_features = self.mlp2(batch_features)
            pair_interactions.index_add_(0, batch_nodes, processed_features)

        pair_interactions = 2 * pair_interactions
        
        return pair_interactions

    def message(self, x_j, edge_attr, norm):
        return  norm.view(-1, 1) * (x_j+edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', fn_count = 3):
 
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

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

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'ncgnn':
                self.convs.append(NCGNNConv(emb_dim))
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
        edge_weights[removed_edge_mask] = 0.00001
    
        return edge_weights

    def forward(self, batched_data, p=10):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x = self.atom_encoder(x)
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
            edge_weights = self.remove_top_k_neighbors_with_mask(edge_index, kappa, p *(layer))
            h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_weights)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation, curv_loss


