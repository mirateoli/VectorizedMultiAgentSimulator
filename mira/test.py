import torch
from torch import Tensor

list = [(0,0),(1,0)]
tensor = torch.tensor(list)

print("list:", list)
print("tensor:", tensor[1])