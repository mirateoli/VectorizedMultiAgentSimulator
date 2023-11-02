import torch
from torch import Tensor

list = [(0,0),(2,0)]
tensor = torch.tensor(list, dtype=torch.float32)

print("list:", list)
print("tensor:", tensor[1])

v = torch.linalg.vector_norm(tensor,dim=1)
print(v)

sq = torch.sum(
    torch.square(tensor), dim=1)
print(sq)