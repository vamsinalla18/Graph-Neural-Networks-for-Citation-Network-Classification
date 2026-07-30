import torch
from torch_geometric.datasets import Planetoid

print(torch.__version__)
print(torch.backends.mps.is_available())

dataset = Planetoid(root="data", name="Cora")
print(dataset)