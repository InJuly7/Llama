import torch
from torch_lib import create_tensor


# 0, 2, 4, ..., 126
A = torch.arange(0, 128, 2, device="cpu")
print(f"A: {A}")

B = A[: (128 // 2)]
print(f"B: {B}")

C = B.float() / 128
print(f"C: {C}")

D = torch.arange(64, device="cpu")
print(f"D: {D}")
