# ============= 场景3: LazyCache（延迟应用RoPE）=============
class LazyCache:
    """
    有些实现会在 Cache 内部应用 RoPE,而不是外部
    这样做的好处是可以节省内存（存储未旋转的 key)
    """

    def __init__(self):
        self.key_cache_unrotated = []  # 存储未应用 RoPE 的 key
        self.value_cache = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        这里需要 sin/cos 在内部应用 RoPE!
        """
        sin = cache_kwargs["sin"]  # 必须的！
        cos = cache_kwargs["cos"]  # 必须的！

        # 存储未旋转的 key（节省存储）
        if layer_idx >= len(self.key_cache_unrotated):
            self.key_cache_unrotated.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache_unrotated[layer_idx] = torch.cat([self.key_cache_unrotated[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # 返回时才应用 RoPE（这里简化了）
        # 实际使用时会对整个序列应用不同位置的 RoPE
        rotated_key = self.apply_rope(self.key_cache_unrotated[layer_idx], sin, cos)

        return rotated_key, self.value_cache[layer_idx]

    def apply_rope(self, x, sin, cos):
        # 简化版 RoPE 应用
        return x  # 实际会做旋转


print("\n测试2: StaticCache (必须要 cache_position)")
static_cache = StaticCache(max_batch_size=batch_size, num_heads=num_heads, max_seq_len=2048, head_dim=head_dim)

# 第一次写入位置 0
cache_kwargs = {"cache_position": torch.tensor([0]), "sin": sin_full[0:1], "cos": cos_full[0:1]}
k, v = static_cache.update(new_key, new_value, 0, cache_kwargs)
print(f"StaticCache 写入位置 {cache_kwargs['cache_position'].item()}, 返回长度: {k.shape[2]}")

# 第二次写入位置 1
cache_kwargs = {"cache_position": torch.tensor([1]), "sin": sin_full[1:2], "cos": cos_full[1:2]}
k, v = static_cache.update(new_key, new_value, 0, cache_kwargs)
print(f"StaticCache 写入位置 {cache_kwargs['cache_position'].item()}, 返回长度: {k.shape[2]}")


print("\n" + "=" * 60)
print("总结:")
print("=" * 60)
print("1. DynamicCache: cache_kwargs 基本没用，RoPE 在外部应用")
print("2. StaticCache: 需要 cache_position 定位写入位置")
print("3. LazyCache: 需要 sin/cos 在内部延迟应用 RoPE")
print("\n统一接口设计：为了支持多种 Cache 类型，")
print("所以传递完整的 cache_kwargs，让各 Cache 自己决定用哪些！")
