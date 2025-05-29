import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, softmax
import math
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple, Union
from collections import defaultdict

import torch.nn.functional as F
from torch import Tensor


class GNNLayer(nn.Module):
    
    def __init__(self, in_features, out_features, params) -> None:
        super(GNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.scalar = params['scalar']
        self.combination = params['combination']


        if self.scalar:
            self.eps = nn.ParameterDict()

        self.transform = nn.ModuleDict()


        self.linear = nn.Linear(self.in_features, self.out_features)

        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def forward(self, h, pair_info):

        # transform roots
        h3 = self.linear(h)

        pairs, degrees, scatter = pair_info

        for key in pairs:
            if len(scatter[key]) == 0:
                continue

            k = str(key)
            
            if self.combination == "multi":  # s(h_x @ W) * s(h_y @ W)
                h_temp = 1
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i]))
                    h_temp = h_temp * self.transform[k](h_t)
            elif self.combination == "sum":  # s(h_x @ W + h_y @ W)
                h_temp = 0
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i]))
                    h_temp = h_temp + h_t
                h_temp = self.transform[k](h_temp)

            h_sum = torch.zeros((h.shape[0], self.out_features)).to(self.dummy_param.device)
            scatter_add(src=h_temp, out=h_sum, index=scatter[key], dim=0)

            if self.scalar:
                h_sum = (1 + self.eps[k]) * h_sum

            h3 = h3 + h_sum

        return h3
    
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return  F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out
 
    
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index):
        x = self.linear(x)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j):
        return F.relu(x_j)

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

    def forward(self, x, edge_index):
        neighbor_sum = self.propagate(edge_index, x=x)
        

        pair_interactions = self.compute_pair_interactions(x, edge_index)
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

    def message(self, x_j):
        return  x_j

    def update(self, aggr_out):
        return aggr_out

class GraphSN(MessagePassing):
    def __init__(self, emb_dim):
        super(GraphSN, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), 
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        
        self.eps = torch.nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, x, edge_index, norm_edge_weight, norm_self_loop):
        out = self.mlp(self.eps * norm_self_loop.view(-1,1) * x + self.propagate(edge_index, x=x, norm=norm_edge_weight))
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

