# BEC-GNN

KDD 2025 Paper - Depth-Adaptive Graph Neural Networks via Learnable Bakry-Émery Curvature

# Python Version

Python: 3.11.5

# PyTorch Libraries

torch: 2.3.1
torchvision: 0.18.1
torchaudio: 2.3.1
torch-geometric: 2.7.0
torch-cluster: 1.6.3
torch-scatter: 2.0.9
torch_sparse: 0.6.18

For TU datasets, you may go into Graph_classification/TU folder and execute,
```
python3 graph_classification.py 
```

For ogbg-moltox21, you may go into Graph_classification/OGB folder and execute,
```
python3 main_pyg.py 
```

For ZINC, you may go into Graph_regression folder and execute,
```
python3 zinc.py 
```

For vertex classification, you may go into Vertex_classification folder and execute,
```
python3 random_splits.py 
```

For ogbn-arxiv, you may go into Vertex_classification folder and execute,
```
python3 ogbn_arxiv.py 
```

For influence estimation, you may go into Influence_estimation folder and execute,
```
python3 estimation.py 
```
