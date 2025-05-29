

import argparse
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger

import time

def get_edge_index(adj_matrix):
    device = adj_matrix.device
    
    if adj_matrix.layout == torch.sparse_csr:
        crow_indices = adj_matrix.crow_indices().to(device)
        col_indices = adj_matrix.col_indices().to(device)
        
        rows = torch.repeat_interleave(
            torch.arange(len(crow_indices)-1, device=device), 
            crow_indices[1:] - crow_indices[:-1]
        ).to(device)
        cols = col_indices.to(device)        
        edge_index = torch.stack([rows, cols], dim=0).to(device)   
        reverse_edge_index = torch.stack([cols, rows], dim=0).to(device)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        edge_index = torch.unique(edge_index, dim=1)      
        return edge_index
    else:
        rows, cols = torch.where(adj_matrix > 0)
        edge_index = torch.stack([rows, cols], dim=0).to(device)
        reverse_edge_index = torch.stack([cols, rows], dim=0).to(device) 
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        edge_index = torch.unique(edge_index, dim=1)        
        return edge_index


def get_edge_index_v1(adj_matrix):
    device = adj_matrix.device 
    
    if adj_matrix.layout == torch.sparse_csr:
        crow_indices = adj_matrix.crow_indices().to(device)
        col_indices = adj_matrix.col_indices().to(device)
        
        rows = torch.repeat_interleave(
            torch.arange(len(crow_indices)-1, device=device), 
            crow_indices[1:] - crow_indices[:-1]
        ).to(device)
        cols = col_indices.to(device)
        edge_index = torch.stack([rows, cols], dim=0).to(device)        
        return edge_index
    else:
        rows, cols = torch.where(adj_matrix > 0)
        return torch.stack([rows, cols], dim=0).to(device)

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
    result = result       
    weighted_diffs = weighted_diffs
    result.scatter_add_(0, src, weighted_diffs)
    return 0.5 * result


def compute_gamma2_optimized(f, edge_index, weights):
    n_vertices = f.size(0)
    src, dst = edge_index[0], edge_index[1]
    f_diff = f[dst] - f[src]
    delta_f = torch.zeros_like(f)
    weighted_diffs = weights * f_diff
    weighted_diffs = weighted_diffs
    delta_f.scatter_add_(0, src, weighted_diffs)
    
    gamma_f = torch.zeros_like(f)
    weighted_squared_diffs = weights * f_diff.pow(2)
    weighted_squared_diffs = weighted_squared_diffs
    gamma_f.scatter_add_(0, src, weighted_squared_diffs)
    gamma_f = 0.5 * gamma_f
    
    gamma_diff = gamma_f[dst] - gamma_f[src]
    delta_gamma = torch.zeros_like(f)
    gamma_diff = gamma_diff
    weighted_gamma_diffs = weights * gamma_diff
    weighted_gamma_diffs = weighted_gamma_diffs
    delta_gamma.scatter_add_(0, src, weighted_gamma_diffs)
    
    delta_f_diff = (delta_f[dst] - delta_f[src])
    gamma_f_delta = torch.zeros_like(f)
    weighted_cross_diffs = weights * f_diff * delta_f_diff
    gamma_f_delta.scatter_add_(0, src, weighted_cross_diffs)
    gamma_f_delta = 0.5 * gamma_f_delta

    gamma2 = 0.5*delta_gamma - gamma_f_delta
    
    return gamma2


def compute_loss(kappa, gamma, gamma2):
    kappa_gamma = kappa.squeeze(-1) * gamma
    diff = kappa_gamma - gamma2
    loss_per_node = torch.relu(diff)
    total_loss = loss_per_node.sum() - kappa.sum()   
    return total_loss




class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, fn_count=2):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels,  cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(in_channels, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(in_channels, 20))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = min(int(n_vertices * k / 100), 100)
        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)
        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0
        src, dst = edge_index[0], edge_index[1]
        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        modified_edge_index = edge_index[:, ~removed_edge_mask]
        return modified_edge_index

    def forward(self, x, adj_t, p=10):
        
        curv_loss = 0
        kappa = self.curv_mlp(x)
        weights = self.weight_model((torch.ones(adj_t.size(1)).float()))
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, adj_t, weights)
            gamma2_f = compute_gamma2_optimized(f, adj_t, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            
        for i, conv in enumerate(self.convs[:-1]):
            adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i))
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i+1))
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1), curv_loss

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, fn_count=2):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels,  cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels, cached=False))
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(in_channels, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(in_channels, 20))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = min(int(n_vertices * k / 100), 100)
        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)
        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0
        src, dst = edge_index[0], edge_index[1]
        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        modified_edge_index = edge_index[:, ~removed_edge_mask]
        return modified_edge_index

    def forward(self, x, adj_t, p=10):
        
        curv_loss = 0
        kappa = self.curv_mlp(x)
        weights = self.weight_model((torch.ones(adj_t.size(1)).float()))
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, adj_t, weights)
            gamma2_f = compute_gamma2_optimized(f, adj_t, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            
        for i, conv in enumerate(self.convs[:-1]):
            adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i))
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i+1))
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1), curv_loss

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, fn_count=2):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels,  cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, cached=False))
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(in_channels, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(in_channels, 20))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = min(int(n_vertices * k / 100), 100)
        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)
        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0
        src, dst = edge_index[0], edge_index[1]
        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        modified_edge_index = edge_index[:, ~removed_edge_mask]
        return modified_edge_index

    def forward(self, x, adj_t, p=10):
        
        curv_loss = 0
        kappa = self.curv_mlp(x)
        weights = self.weight_model((torch.ones(adj_t.size(1)).float()))
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, adj_t, weights)
            gamma2_f = compute_gamma2_optimized(f, adj_t, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            
        for i, conv in enumerate(self.convs[:-1]):
            adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i))
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i+1))
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1), curv_loss
    
    
class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, fn_count=2):
        super(SGC, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SGConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SGConv(hidden_channels, hidden_channels,  cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SGConv(hidden_channels, out_channels, cached=False))
        self.weight_model = WeightMLP()
        self.curv_mlp = CurvatureMLP(in_channels, 20)
        self.fn_mlp = []
        for i in range(fn_count):
            self.fn_mlp.append(CurvatureMLP(in_channels, 20))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def remove_top_k_neighbors_with_mask(self, edge_index, curvatures, k):
        n_vertices = curvatures.size(0)
        num_to_remove = min(int(n_vertices * k / 100), 100)
        top_values, top_indices = torch.topk(curvatures.squeeze(-1), num_to_remove)
        vertex_mask = torch.ones(n_vertices, device=edge_index.device)
        vertex_mask[top_indices] = 0
        src, dst = edge_index[0], edge_index[1]
        removed_edge_mask = (vertex_mask[src] == 0) | (vertex_mask[dst] == 0)
        modified_edge_index = edge_index[:, ~removed_edge_mask]
        return modified_edge_index

    def forward(self, x, adj_t, p=10):
        
        curv_loss = 0
        kappa = self.curv_mlp(x)
        weights = self.weight_model((torch.ones(adj_t.size(1)).float()))
        for i in self.fn_mlp:
            f = i(x.float()).squeeze(-1)
            gamma_f = compute_gamma(f, adj_t, weights)
            gamma2_f = compute_gamma2_optimized(f, adj_t, weights)
            curv_loss += compute_loss(kappa, gamma_f, gamma2_f)
            
        for i, conv in enumerate(self.convs[:-1]):
            adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i))
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        adj_t = self.remove_top_k_neighbors_with_mask(adj_t, kappa.squeeze(-1), p * (i+1))
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1), curv_loss

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out, curv_loss = model(data.x,get_edge_index( data.adj_t))
    out = out[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx]) + curv_loss
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out,curv_loss = model(data.x, get_edge_index(data.adj_t))
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    start = time.time()

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    end = time.time()
    print(f"Execution time: {(end - start) * 1000:.3f} ms")
    logger.print_statistics()


if __name__ == "__main__":
    main()
