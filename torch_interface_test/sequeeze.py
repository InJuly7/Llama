import torch

A = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("原始形状:", A.shape)  # torch.Size([4])
print("原始数据:\n", A)

# 应用 [None, :, None] 操作
result = A[None, :, None]
print("\n变换后形状:", result.shape)  # torch.Size([1, 4, 1])
print("变换后数据:\n", result)
