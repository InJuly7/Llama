import torch

print("=" * 50)
print("示例1: 基础用法")
print("=" * 50)

# 创建一个形状为 (2, 4, 2) 的实数张量
# 最后一维的2表示 [实部, 虚部]
real_tensor = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]]  # 第一个样本  # 第二个样本
)

print(f"原始实数张量形状: {real_tensor.shape}")
print(f"原始张量:\n{real_tensor}\n")

# 转换为复数张量
complex_tensor = torch.view_as_complex(real_tensor)

print(f"复数张量形状: {complex_tensor.shape}")
print(f"复数张量:\n{complex_tensor}\n")

print("=" * 50)
print("示例2: 在RoPE中的典型应用")
print("=" * 50)

# 模拟 RoPE 中的查询向量
batch_size, seq_len, num_heads, head_dim = 2, 3, 4, 8

# 假设 xq 是查询向量，形状为 (batch, seq_len, num_heads, head_dim)
xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
print(f"原始查询向量 xq 形状: {xq.shape}")

# reshape成 (batch, seq_len, num_heads, head_dim//2, 2)
# 将最后一维分成两半，准备转换为复数
xq_ = xq.reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)
print(f"reshape后形状: {xq_.shape}")

# 转换为复数形式
xq_complex = torch.view_as_complex(xq_)
print(f"复数形式形状: {xq_complex.shape}")
print(f"数据类型变化: {xq.dtype} -> {xq_complex.dtype}\n")

print("=" * 50)
print("示例3: 逆操作 - view_as_real()")
print("=" * 50)

# 可以用 view_as_real 转换回去
xq_back = torch.view_as_real(xq_complex)
print(f"转换回实数形状: {xq_back.shape}")
print(f"与原始reshape后是否相同: {torch.allclose(xq_, xq_back)}")


"""
Output Log:
==================================================
示例1: 基础用法
==================================================
原始实数张量形状: torch.Size([2, 4, 2])
原始张量:
tensor([[[ 1.,  2.],
         [ 3.,  4.],
         [ 5.,  6.],
         [ 7.,  8.]],

        [[ 9., 10.],
         [11., 12.],
         [13., 14.],
         [15., 16.]]])

复数张量形状: torch.Size([2, 4])
复数张量:
tensor([[ 1.+2.j,  3.+4.j,  5.+6.j,  7.+8.j],
        [ 9.+10.j, 11.+12.j, 13.+14.j, 15.+16.j]])

==================================================
示例2: 在RoPE中的典型应用
==================================================
原始查询向量 xq 形状: torch.Size([2, 3, 4, 8])
reshape后形状: torch.Size([2, 3, 4, 4, 2])
复数形式形状: torch.Size([2, 3, 4, 4])
数据类型变化: torch.float32 -> torch.complex64

==================================================
示例3: 逆操作 - view_as_real()
==================================================
转换回实数形状: torch.Size([2, 3, 4, 4, 2])
与原始reshape后是否相同: True
"""
