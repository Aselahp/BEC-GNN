import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F
import math

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(
            Linear(input_dim, hidden_dim), 
            Dropout(dropout),
            ReLU(), 
            BatchNorm1d(batchnorm_dim),
            Linear(hidden_dim, hidden_dim), 
            Dropout(dropout),
            ReLU(), 
            BatchNorm1d(batchnorm_dim)
        )
        
        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        
    def forward(self, A, X):
        batch, N = A.shape[:2]
        identity = torch.eye(N, device=A.device).unsqueeze(0).expand(batch, -1, -1)
        A_gin = (1 + self.eps) * identity + A
        X = A_gin @ X
        X = self.mlp(X)        
        return X

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(
            Linear(input_dim, hidden_dim),
            Dropout(dropout),
            ReLU(),
            BatchNorm1d(batchnorm_dim),
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout),
            ReLU(),
            BatchNorm1d(batchnorm_dim)
        )
        
    def forward(self, A, X):
        batch, N = A.shape[:2]
        identity = torch.eye(N, device=A.device).unsqueeze(0).expand(batch, -1, -1)
        A_hat = A + identity
        D = torch.sum(A_hat, dim=2, keepdim=True)
        D_inv_sqrt = torch.pow(D + 1e-7, -0.5)
        A_norm = D_inv_sqrt * A_hat * D_inv_sqrt.transpose(1, 2)
        X = A_norm @ X
        X = self.mlp(X)        
        return X

class GraphSN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout), 
                              ReLU(), BatchNorm1d(batchnorm_dim),
                              Linear(hidden_dim, hidden_dim), Dropout(dropout), 
                              ReLU(), BatchNorm1d(batchnorm_dim))
        
        self.linear = Linear(hidden_dim, hidden_dim)
        
        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, A, X):
        batch, N = A.shape[:2]
        mask = torch.eye(N).unsqueeze(0)
        batch_diagonal = torch.diagonal(A, 0, 1, 2)
        batch_diagonal = self.eps * batch_diagonal
        A = mask*torch.diag_embed(batch_diagonal) + (1. - mask)*A
        X = self.mlp(A @ X)
        X = self.linear(X)
        X = F.relu(X)        
        return X
    
class NCGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp1 = Sequential(
            Linear(input_dim, hidden_dim), 
            Dropout(dropout),
            ReLU(), 
            BatchNorm1d(batchnorm_dim),
            Linear(hidden_dim, hidden_dim), 
            Dropout(dropout),
            ReLU(), 
            BatchNorm1d(batchnorm_dim)
        )
        
        self.mlp2 = Sequential(
            Linear(input_dim, hidden_dim),
            Dropout(dropout),
            ReLU(),
            Linear(hidden_dim, input_dim),
            ReLU()
        )
        
        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        
    def forward(self, A, X):
        batch, N = A.shape[:2]
        identity = torch.eye(N, device=A.device).unsqueeze(0).expand(batch, -1, -1)
        A_standard = (1 + self.eps) * identity + A
        X_standard = A_standard @ X
        X_pairwise = torch.zeros_like(X)

        for b in range(batch):
            for i in range(N):
                neighbors = torch.nonzero(A[b, i]).squeeze(1)
                if len(neighbors.shape) == 0:
                    continue
                pair_features = []
                for j_idx in range(len(neighbors)):
                    for k_idx in range(j_idx + 1, len(neighbors)):
                        j, k = neighbors[j_idx], neighbors[k_idx]
                        if A[b, j, k]:
                            pair_sum = X[b, j] + X[b, k]
                            pair_features.append(pair_sum)
                
                if pair_features: 
                    stacked_features = torch.stack(pair_features)
                    transformed_features = self.mlp2(stacked_features)
                    X_pairwise[b, i] = transformed_features.sum(dim=0)
        
        X = X_standard + X_pairwise
        X = self.mlp1(X)
        
        return X
    
