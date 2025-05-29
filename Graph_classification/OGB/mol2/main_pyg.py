

import torch
from torch_geometric.data import DataLoader, Data
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx

from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
from torch_geometric.utils import to_networkx

from time import process_time

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

K = 3

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, curv_loss = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]) + curv_loss
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]) + curv_loss
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, curv_loss = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def compute_augmented_features_onehot(features, max_value):

    N = features.shape[0]
    K = features.shape[1]

    one_hot_features = np.zeros((N, K * (max_value + 1)))

    for i in range(N):
        for k in range(K):
            value = int(features[i, k])
            one_hot_index = k * (max_value + 1) + value
            one_hot_features[i, one_hot_index] = 1

    return one_hot_features

def compute_augmented_features(adj, K):
    assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square."

    N = adj.shape[0]
    augmented_features = np.zeros((N, K))
    Ak = np.eye(N)

    for k in range(1, K + 1):
        Ak = np.dot(Ak, adj)
        cycle_counts = np.diag(Ak)
        augmented_features[:, k - 1] = cycle_counts

    return augmented_features

def add_cycle_counts(dataset):
    data_list = []
    dataset_length = len(dataset)
    for itr in np.arange(dataset_length):
        row, col = dataset[itr]['edge_index']
        num_of_nodes = dataset[itr]['x'].shape[0]
        adj = torch.zeros(num_of_nodes, num_of_nodes)
        for i in np.arange(row.shape[0]):
            adj[row[i]][col[i]]=1.0

        A_array = adj.detach().numpy()
        G = nx.from_numpy_matrix(A_array)   
        augmented_features = torch.tensor(compute_augmented_features(adj, K))
        data = Data(edge_attr=dataset[itr]['edge_attr'], edge_index=dataset[itr]['edge_index'], x=dataset[itr]['x'], 
                    y=dataset[itr]['y'])
        data.c = augmented_features
        data_list.append(data)  
    return data_list

def add_structural_coefficients(dataset):
    data_list = []
    dataset_length = len(dataset)
    for itr in np.arange(dataset_length):
        row, col = dataset[itr]['edge_index']
        num_of_nodes = dataset[itr]['x'].shape[0]
        adj = torch.zeros(num_of_nodes, num_of_nodes)
        for i in np.arange(row.shape[0]):
            adj[row[i]][col[i]]=1.0

        A_array = adj.detach().numpy()
        G = nx.from_numpy_matrix(A_array)
    
        sub_graphs = []
        subgraph_nodes_list = []
        sub_graphs_adj = []
        sub_graph_edges = []
        new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])

        for i in np.arange(len(A_array)):
            s_indexes = []
            for j in np.arange(len(A_array)):
                s_indexes.append(i)
                if(A_array[i][j]==1):
                    s_indexes.append(j)
            sub_graphs.append(G.subgraph(s_indexes))

        for i in np.arange(len(sub_graphs)):
            subgraph_nodes_list.append(list(sub_graphs[i].nodes))

        for index in np.arange(len(sub_graphs)):
            sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())

        for index in np.arange(len(sub_graphs)):
            sub_graph_edges.append(sub_graphs[index].number_of_edges())

        for node in np.arange(len(subgraph_nodes_list)):
            sub_adj = sub_graphs_adj[node]
            for neighbors in np.arange(len(subgraph_nodes_list[node])):
                index = subgraph_nodes_list[node][neighbors]
                count = torch.tensor(0).float()
                if(index==node):
                    continue
                else:
                    c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                    if index in c_neighbors:
                        nodes_list = subgraph_nodes_list[node]
                        sub_graph_index = nodes_list.index(index)
                        c_neighbors_list = list(c_neighbors)
                        for i, item1 in enumerate(nodes_list):
                            if(item1 in c_neighbors):
                                for item2 in c_neighbors_list:
                                    j = nodes_list.index(item2)
                                    count += sub_adj[i][j]

                    new_adj[node][index] = count/2
                    new_adj[node][index] = new_adj[node][index]/(len(c_neighbors)*(len(c_neighbors)-1))
                    new_adj[node][index] = new_adj[node][index] * (len(c_neighbors)**2)

        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)
        weight = weight + torch.FloatTensor(A_array)

        weight = weight.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)
        weight = torch.FloatTensor(weight)
        coeff = weight.sum(1, keepdim=True)
        coeff = (coeff.T)[0]

        weight = weight.detach().numpy()

        data = Data(edge_attr=dataset[itr]['edge_attr'], edge_index=dataset[itr]['edge_index'], x=dataset[itr]['x'], 
                    y=dataset[itr]['y'])
        data.norm_edge_weight = torch.FloatTensor(weight[np.nonzero(weight)])
        data.norm_self_loop = torch.FloatTensor(coeff)

        data_list.append(data)

    return data_list


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='graphSN',
                        help='GNN graphSN, graphSN-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=200,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-moltox21",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args('')

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    # structural coefficients
    data_list = add_structural_coefficients(dataset)
    
    # ID-GNN
    #data_list = add_cycle_counts(dataset)
    
    split_idx = dataset.get_idx_split()

    train_data_list = []
    valid_data_list = []
    test_data_list = []

    for i in split_idx['train']:
        index = torch.IntTensor.item(i)
        train_data_list.append(data_list[index])

    for i in split_idx['valid']:
        index = torch.IntTensor.item(i)
        valid_data_list.append(data_list[index])

    for i in split_idx['test']:
        index = torch.IntTensor.item(i)
        test_data_list.append(data_list[index])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data_list, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    val_roc = []
    test_roc = []
    
    for i in range(2):
        if args.gnn == 'graphSN':
            model = GNN(gnn_type = 'graphSN', num_tasks = dataset.num_tasks, num_layer = args.num_layer, 
                        emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gin':
            model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, 
                        emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train(model, device, train_loader, optimizer, dataset.task_type)

            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

        if 'classification' in dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)

        print('Finished training!')
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        val_roc.append(valid_curve[best_val_epoch])
        print('Test score: {}'.format(test_curve[best_val_epoch]))
        test_roc.append(test_curve[best_val_epoch])

        if not args.filename == '':
            torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)
    
    average = np.mean(val_roc)
    standard_dev = np.std(val_roc)
    print('Mean validation avg ROC (+- std): {} ({})'.format(average, standard_dev))
    
    average = np.mean(test_roc)
    standard_dev = np.std(test_roc)
    print('Mean test avg ROC (+- std): {} ({})'.format(average, standard_dev))
    print(test_roc)


if __name__ == "__main__":
    main()