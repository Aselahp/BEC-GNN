import torch
import math

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid,Coauthor,Amazon,WikipediaNetwork, DBLP, WebKB, Actor
        
def DataLoader(name):
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=None)
    elif name in ['cs', 'physics']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Coauthor(path, name, transform=None)
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, transform=None)
    elif name in ['chameleon', 'crocodile', 'squirrel']:
         root_path = './'
         path = osp.join(root_path, 'data', name)
         dataset = WikipediaNetwork(path, name, transform=None)
    elif name in ['cornell', 'texas', 'wisconsin']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = WebKB(path, name)
    elif name in ['dblp']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = DBLP(path, name)
    elif name == 'actor':  # Custom Actor dataset
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Actor(path)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')


    return dataset