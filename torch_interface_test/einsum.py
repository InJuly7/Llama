import torch

# ============= 简单示例 =============
print("=" * 50)
print("示例1: 理解 einsum 的外积操作")
print("=" * 50)

# 假设我们有4个位置
position_ids = torch.tensor([0, 1, 2, 3])  # shape: [4]

# 假设我们有3个频率维度
inv_freq = torch.tensor([1.0, 0.1, 0.01])  # shape: [3]

# 使用 einsum 计算外积
freqs = torch.einsum("i,j->ij", position_ids.float(), inv_freq)

print(f"position_ids shape: {position_ids.shape}")
print(f"inv_freq shape: {inv_freq.shape}")
print(f"freqs shape: {freqs.shape}")
print(f"\nfreqs 结果:\n{freqs}")

# ============= 等价操作 =============
print("\n" + "=" * 50)
print("示例2: einsum 等价于外积")
print("=" * 50)

# 方法1: einsum
freqs_einsum = torch.einsum("i,j->ij", position_ids.float(), inv_freq)

# 方法2: 使用 outer
freqs_outer = torch.outer(position_ids.float(), inv_freq)

# 方法3: 使用广播
freqs_broadcast = position_ids.float().unsqueeze(1) * inv_freq.unsqueeze(0)

print(f"三种方法结果相同: {torch.allclose(freqs_einsum, freqs_outer) and torch.allclose(freqs_einsum, freqs_broadcast)}")

# ============= 实际RoPE场景 =============
print("\n" + "=" * 50)
print("示例3: 模拟真实的RoPE场景")
print("=" * 50)


class SimpleRoPE:
    def __init__(self, dim=4, max_position=10):
        """
        dim: 嵌入维度（必须是偶数）
        max_position: 最大位置数
        """
        self.dim = dim
        # 计算逆频率: 1 / (10000 ^ (2i / dim))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq
        print(f"inv_freq shape: {inv_freq.shape}")
        print(f"inv_freq: {inv_freq}")

    def forward(self, position_ids):
        """
        position_ids: [batch_size, seq_len] or [seq_len]
        """
        # 计算位置编码
        freqs = torch.einsum("i,j->ij", position_ids.flatten().float(), self.inv_freq)
        print(f"\nfreqs shape: {freqs.shape}")
        print(f"freqs:\n{freqs}")

        # 通常还会继续计算 sin 和 cos
        emb = torch.cat([freqs, freqs], dim=-1)  # 复制一份用于实部和虚部
        cos_emb = emb.cos()
        sin_emb = emb.sin()

        return cos_emb, sin_emb


# 创建RoPE实例
rope = SimpleRoPE(dim=4, max_position=10)

# 假设batch_size=2, seq_len=3
position_ids = torch.arange(0, 3)  # [0, 1, 2]

cos_emb, sin_emb = rope.forward(position_ids)
print(f"\ncos_emb shape: {cos_emb.shape}")
print(f"sin_emb shape: {sin_emb.shape}")

# ============= 可视化理解 =============
print("\n" + "=" * 50)
print("示例4: 可视化理解每个元素的计算")
print("=" * 50)

pos = torch.tensor([0, 1, 2]).float()
freq = torch.tensor([1.0, 0.5]).float()

result = torch.einsum("i,j->ij", pos, freq)

print("计算过程:")
for i, p in enumerate(pos):
    for j, f in enumerate(freq):
        print(f"result[{i},{j}] = pos[{i}] * freq[{j}] = {p:.1f} * {f:.1f} = {result[i,j]:.1f}")

print(f"\n最终结果:\n{result}")
