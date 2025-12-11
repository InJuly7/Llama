import torch


class StaticCache:
    """每一层都会创建自己的StaticCache实例"""

    def __init__(self, max_batch_size, num_heads, max_seq_len, head_dim):
        self.key_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.value_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        cache_position = cache_kwargs["cache_position"]

        start_pos = cache_position[0].item()
        seq_len = key_states.shape[2]
        end_pos = start_pos + seq_len

        # 写入指定位置
        self.key_cache[:, :, start_pos:end_pos, :] = key_states
        self.value_cache[:, :, start_pos:end_pos, :] = value_states

        return (self.key_cache[:, :, :end_pos, :], self.value_cache[:, :, :end_pos, :])


# =================== 完整Demo ===================
class TransformerWithStaticCache:
    def __init__(self, num_layers=3, num_heads=8, head_dim=64, max_seq_len=100):
        self.num_layers = num_layers

        # 🔥关键：每一层都有自己独立的Cache实例
        self.caches = [
            StaticCache(max_batch_size=1, num_heads=num_heads, max_seq_len=max_seq_len, head_dim=head_dim) for _ in range(num_layers)
        ]

    def forward(self, input_ids, cache_position):
        """
        cache_position: 告诉每一层, 当前token应该写入cache的哪个位置
        """
        batch_size, seq_len = input_ids.shape
        num_heads = 8
        head_dim = 64

        # 模拟每一层的处理
        for layer_idx in range(self.num_layers):
            print(f"\n--- Layer {layer_idx} ---")

            # 模拟当前层计算出的key/value
            key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
            value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

            # 🔥 每层使用自己的cache
            cache_kwargs = {"cache_position": cache_position}
            cached_key, cached_value = self.caches[layer_idx].update(key_states, value_states, layer_idx, cache_kwargs)

            print(f"Input key shape: {key_states.shape}")
            print(f"Cached key shape: {cached_key.shape}")
            print(f"Cache position: {cache_position}")


# =================== 使用示例 ===================
model = TransformerWithStaticCache(num_layers=3)

print("=" * 60)
print("阶段1: Prefill - 处理prompt 'Hello world' (假设5个token)")
print("=" * 60)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # shape: [1, 5]
cache_position = torch.arange(0, 5)  # [0, 1, 2, 3, 4]
model.forward(input_ids, cache_position)

print("\n" + "=" * 60)
print("阶段2: Decode - 生成第1个token (位置5)")
print("=" * 60)
new_token = torch.tensor([[6]])  # shape: [1, 1]
cache_position = torch.tensor([5])  # 写入位置5
model.forward(new_token, cache_position)

print("\n" + "=" * 60)
print("阶段3: Decode - 生成第2个token (位置6)")
print("=" * 60)
new_token = torch.tensor([[7]])
cache_position = torch.tensor([6])  # 写入位置6
model.forward(new_token, cache_position)

"""
Output Log:
============================================================
阶段1: Prefill - 处理prompt 'Hello world' (假设5个token)
============================================================

--- Layer 0 ---
Input key shape: torch.Size([1, 8, 5, 64])
Cached key shape: torch.Size([1, 8, 5, 64])
Cache position: tensor([0, 1, 2, 3, 4])

--- Layer 1 ---
Input key shape: torch.Size([1, 8, 5, 64])
Cached key shape: torch.Size([1, 8, 5, 64])
Cache position: tensor([0, 1, 2, 3, 4])

--- Layer 2 ---
Input key shape: torch.Size([1, 8, 5, 64])
Cached key shape: torch.Size([1, 8, 5, 64])
Cache position: tensor([0, 1, 2, 3, 4])

============================================================
阶段2: Decode - 生成第1个token (位置5)
============================================================

--- Layer 0 ---
Input key shape: torch.Size([1, 8, 1, 64])
Cached key shape: torch.Size([1, 8, 6, 64])
Cache position: tensor([5])

--- Layer 1 ---
Input key shape: torch.Size([1, 8, 1, 64])
Cached key shape: torch.Size([1, 8, 6, 64])
Cache position: tensor([5])

--- Layer 2 ---
Input key shape: torch.Size([1, 8, 1, 64])
Cached key shape: torch.Size([1, 8, 6, 64])
Cache position: tensor([5])

============================================================
阶段3: Decode - 生成第2个token (位置6)
============================================================

--- Layer 0 ---
Input key shape: torch.Size([1, 8, 1, 64])
Cached key shape: torch.Size([1, 8, 7, 64])
Cache position: tensor([6])

--- Layer 1 ---
Input key shape: torch.Size([1, 8, 1, 64])
Cached key shape: torch.Size([1, 8, 7, 64])
Cache position: tensor([6])

--- Layer 2 ---
Input key shape: torch.Size([1, 8, 1, 64])
Cached key shape: torch.Size([1, 8, 7, 64])
Cache position: tensor([6])
"""
