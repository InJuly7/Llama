import torch

# 模拟场景：假设这是 attention 中的 query 和 key
batch_size = 2
seq_len = 3
num_heads = 4
head_dim = 8  # 必须是偶数

# 原始张量
xq = torch.randn(batch_size, num_heads, seq_len, head_dim)
xk = torch.randn(batch_size, num_heads, seq_len, head_dim)

print("原始形状:")
print(f"xq.shape: {xq.shape}")  # torch.Size([2, 4, 3, 8])
print(f"xk.shape: {xk.shape}")  # torch.Size([2, 4, 3, 8])

# print(xq.shape)
# print(*xq.shape)
# print(xq.shape[:-1])
# print(*xq.shape[:-1])

# 核心操作：将最后一维拆分成 (head_dim/2, 2)
xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

print("\n变换后形状:")
print(f"xq_.shape: {xq_.shape}")  # torch.Size([2, 4, 3, 4, 2])
print(f"xk_.shape: {xk_.shape}")  # torch.Size([2, 4, 3, 4, 2])

# 详细解释
print("\n详细示例:")
print("原始 xq 最后一维的前4个值:", xq[0, 0, 0, :4])
print("重塑后对应位置:", xq_[0, 0, 0, :2])

"""
Output Log:
原始形状:
xq.shape: torch.Size([2, 4, 3, 8])
xk.shape: torch.Size([2, 4, 3, 8])

变换后形状:
xq_.shape: torch.Size([2, 4, 3, 4, 2])
xk_.shape: torch.Size([2, 4, 3, 4, 2])

详细示例:
原始 xq 最后一维的前4个值: tensor([ 0.1257, -1.3391,  0.5015,  0.3703])
重塑后对应位置: tensor([[ 0.1257, -1.3391],
        [ 0.5015,  0.3703]])
"""
