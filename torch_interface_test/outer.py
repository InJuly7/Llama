import torch
from torch_lib import create_tensor

A = torch.arange(0, 5, device="cpu")
print(f"Shape A: {A.shape}")
print(f"A: {A}")

B = torch.arange(0, 10, device="cpu")
print(f"Shape B: {B.shape}")
print(f"B: {B}")

C = torch.outer(A, B).float()
print(f"Shape C: {C.shape}")
print(f"C: {C}")
