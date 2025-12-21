import torch

# 列表推导式
x = torch.randn(2, 3, 4, 5)
print(f"原始shape: {x.shape}")  # torch.Size([2, 3, 4, 5])

ndim = x.ndim  # 4

# 原始代码
shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
print(f"新shape: {shape}")  # [1, 3, 1, 5]

# 详细解释每一步
print("\n逐步分析:")
for i, d in enumerate(x.shape):
    if i == 1 or i == ndim - 1:
        print(f"索引{i}: 保留维度 {d}")
    else:
        print(f"索引{i}: 压缩为 1 (原本是{d})")

"""
Output Log:
原始shape: torch.Size([2, 3, 4, 5])
新shape: [1, 3, 1, 5]

逐步分析:
索引0: 压缩为 1 (原本是2)
索引1: 保留维度 3
索引2: 压缩为 1 (原本是4)
索引3: 保留维度 5
"""
