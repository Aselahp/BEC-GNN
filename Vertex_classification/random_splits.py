from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from scipy import sparse
from scipy.linalg import fractional_matrix_power

from utils import *
from models import DC_GCN, DC_GraphSAGE, DC_GAT, DC_SGC, DC_MixHop, DC_DirGNN
from dataset_utils import DataLoader

import pickle

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.1,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--early_stopping', type=int, default=1000)
parser.add_argument('--train_rate', type=float, default=0.48)
parser.add_argument('--val_rate', type=float, default=0.32)
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='wisconsin', help='Dataset name.')


args = parser.parse_args("")
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dname = args.dataset
dataset = DataLoader(dname)
data = dataset[0]

train_rate = args.train_rate
val_rate = args.val_rate
percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
val_lb = int(round(val_rate*len(data.y)))

permute_masks = random_planetoid_splits
data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

A_norm, A, X, labels, idx_train, idx_val, idx_test = load_citation_data(data)


features = torch.FloatTensor(X)
labels = torch.LongTensor(labels) 

        
adj = torch.FloatTensor(A)


# Model and optimizer
model = DC_GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,
                iterations = 2, 
                adj = adj
                )

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, curv_loss = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + curv_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
def test():
    model.eval()
    logits, curv_loss = model(features, adj)
    accs, losses, preds = [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = accuracy(logits[mask], labels[mask])
        
        loss = F.nll_loss(logits[mask], labels[mask]) + curv_loss

        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses
Results0 = []
for i in range(10):
    
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test()

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    Results0.append([test_acc, best_val_acc])
    
test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')