

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import copy
import time
import networkx as nx
import random
import pickle
from scipy.special import softmax
from scipy.sparse import csr_matrix
import pandas as pd
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt

from main.utils import load_dataset, InverseProblemDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import math
import scipy.sparse as sp
import statistics
from est_models import DC_GCN, DC_GAT, DC_GraphSAGE

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser(description="InfEs")
datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience']
parser.add_argument("-d", "--dataset", default="jazz", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT']
parser.add_argument("-dm", "--diffusion_model", default="IC", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=10, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))
args = parser.parse_args(args=[])

n = 2 #number of layers


with open('data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)

adj, dataset = graph['adj'], graph['inverse_pairs']


#generate feature matrix
adjacency_matrix = torch.sparse_coo_tensor(
    torch.tensor([adj.tocoo().row, adj.tocoo().col], dtype=torch.long),
    torch.tensor(adj.tocoo().data, dtype=torch.float32),
    torch.Size(adj.tocoo().shape)
)

two_degree = torch.sparse.mm(adjacency_matrix, adjacency_matrix)
three_degree = torch.sparse.mm(two_degree, adjacency_matrix)
degree = (torch.sparse.sum(adjacency_matrix, dim=1) ).to_dense()
unique_degrees = torch.unique(degree)

one_hot_encoder = {deg.item(): i for i, deg in enumerate(unique_degrees)}
num_unique_degrees = len(unique_degrees)
num_nodes = adjacency_matrix.size(0)
feature_matrix = torch.zeros((num_nodes, num_unique_degrees))

for i, deg in enumerate(degree):
    one_hot_index = one_hot_encoder[deg.item()]
    feature_matrix[i, one_hot_index] = 1.0
    

adj = torch.Tensor(adj.toarray()).to_sparse()
adj = adj.to(device)

edge_index = adj.coalesce().indices()


def estimation_loss(y, y_hat):
    forward_loss = F.mse_loss(y_hat.squeeze(), y, reduction='sum')
    return forward_loss 



if args.dataset == 'random5':
    batch_size = 2
else:
    batch_size = 32



kf = KFold(n_splits=10, shuffle=True, random_state=42)
val_error = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}')
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
    
    model = DC_GCN(in_channels=feature_matrix.shape[1], hidden_channels=32, out_channels=1, dropout=0.5, adj=adj, num_layers=n)

    optimizer = Adam([{'params': model.parameters()}], lr=0.01)
    
    # Training loop
    for epoch in range(100):
        begin = time.time()
        total_overall = 0

        for batch_idx, data_pair in enumerate(train_loader):
            
            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)
            
            loss = 0
            for i, x_i in enumerate(x):
                x_i = x[i]
                y_i = y[i]
                
                x_hat = feature_matrix
                y_hat, curv_loss = model(x_hat, edge_index)
                total = estimation_loss(y_i , y_hat) + curv_loss
                            
                loss += total

            total_overall += loss.item()
            loss = loss/x.size(0)
            loss.backward()
            optimizer.step()
            
        end = time.time()
        print("Epoch: {}".format(epoch+1), 
              "\tTotal: {:.4f}".format(total_overall / len(train_subset)),
              "\tTime: {:.4f}".format(end - begin)
             )
        
    val_mae = 0
    with torch.no_grad():
        for batch_idx, data_pair in enumerate(val_loader):
            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)

            x_hat = feature_matrix
            y_hat, curv_loss = model(x_hat, edge_index)
            val_mae += np.abs(y_hat.squeeze() - y[0]).sum()/x[0].shape[0]

                
    
    val_mae /= len(val_loader)
    val_error.append(val_mae)
    print('Validation Loss: ', val_mae)
         
mean = np.mean(val_error)
std_dev = np.std(val_error, ddof=1)  

print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")






