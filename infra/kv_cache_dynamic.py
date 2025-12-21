import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import math


class RotaryPositionEmbedding:
    """旋转位置编码"""

    def __init__(self, head_dim, max_position_embeddings=2048):
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

    def forward(
        self,
    ):
        # 使用随机数 简化流程
        sin = torch.randn(1, self.max_position_embeddings, self.head_dim)  # [B, S, D]
        cos = torch.randn(1, self.max_position_embeddings, self.head_dim)
        return sin, cos


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DynamicCache:
    """动态缓存 - 每次都拼接, RoPE 在外部应用"""

    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx >= len(self.key_cache):
            # 第一次，直接存储
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 拼接
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        """获取当前缓存的序列长度"""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]


class SimpleAttentionLayer:
    """简化的Attention层(只关注KV cache的使用)"""

    def __init__(self, num_heads, head_dim, layer_idx):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,S,H]
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: DynamicCache = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):

        batch_size, seq_len, _ = hidden_states.shape  # [B, S, N, D]

        q = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # transpose: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [B, N, S, D]
        k = k.transpose(1, 2)  # [B, N, S, D]
        v = v.transpose(1, 2)  # [B, N, S, D]

        cos, sin = position_embeddings  # [B,S,D]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. 更新cache，获取完整的K, V
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = cache.update(k, v, self.layer_idx)  # [B,N,S2,D]

        print(f"  Layer {self.layer_idx}:")
        print(f"    当前 Q shape: {q.shape}")
        print(f"    当前 K shape: {k.shape}")
        print(f"    缓存后完整 K shape: {key_states.shape}")
        print(f"    缓存后完整 V shape: {value_states.shape}")

        return hidden_states  # 简化，直接返回


# ==================== 主流程Demo ====================
def demo():
    print("=" * 60)
    print("DynamicCache + RoPE 工作流程演示")
    print("=" * 60)

    # 参数配置
    batch_size = 2
    num_heads = 32
    seq_len = 10  # prefill阶段的序列长度
    head_dim = 64
    max_position_embeddings = 2048
    num_layers = 3  # 模拟3层
    hidden_size = num_heads * head_dim

    # 初始化组件
    rope = RotaryPositionEmbedding(head_dim, max_position_embeddings)
    sin_cache, cos_cache = rope.forward()

    cache = DynamicCache()
    layers = [SimpleAttentionLayer(num_heads, head_dim, i) for i in range(num_layers)]

    print(f"\n配置信息:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")

    # ==================== Prefill 阶段 ====================
    print("\n" + "=" * 60)
    print("【Prefill 阶段】处理完整的prompt序列")
    print("=" * 60)

    # 模拟输入的prompt embeddings
    prompt_embeddings = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\nPrompt embeddings shape: {prompt_embeddings.shape}")

    # position_ids: [batch_size, seq_len]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    print(f"Position IDs shape: {position_ids.shape}")
    print(f"Position IDs: {position_ids[0].tolist()}")

    # 通过所有层
    hidden_states = prompt_embeddings
    position_embeddings = (cos_cache[:, :seq_len, :], sin_cache[:, :seq_len, :])
    for layer in layers:
        hidden_states = layer.forward(hidden_states, position_embeddings=position_embeddings, cache=cache)

    print(f"\nPrefill完成后, KV cache中的序列长度: {cache.get_seq_length()}")

    # ==================== Decode 阶段 ====================
    print("\n" + "=" * 60)
    print("【Decode 阶段】逐个生成新token")
    print("=" * 60)

    num_new_tokens = 5  # 生成5个新token

    for step in range(num_new_tokens):
        print(f"\n--- 生成第 {step + 1} 个新token ---")

        # 模拟新生成的token embedding
        # 注意：decode阶段每次只处理1个token
        new_token_embedding = torch.randn(batch_size, 1, hidden_size)
        print(f"新token embedding shape: {new_token_embedding.shape}")

        # 当前token的position_id
        current_seq_len = cache.get_seq_length()
        position_ids = torch.tensor([[current_seq_len]] * batch_size)
        print(f"当前position ID: {position_ids[0].item()}")

        # 通过所有层
        hidden_states = new_token_embedding
        for layer in layers:
            position_embeddings = (
                cos_cache[:, current_seq_len : current_seq_len + 1, :],
                sin_cache[:, current_seq_len : current_seq_len + 1, :],
            )
            hidden_states = layer.forward(hidden_states, position_embeddings=position_embeddings, cache=cache)

        print(f"当前KV cache总长度: {cache.get_seq_length()}")


if __name__ == "__main__":
    demo()


"""
Output Log:
============================================================
DynamicCache + RoPE 工作流程演示
============================================================

配置信息:
  batch_size: 2
  num_heads: 32
  head_dim: 64
  hidden_size: 2048
  num_layers: 3

============================================================
【Prefill 阶段】处理完整的prompt序列
============================================================

Prompt embeddings shape: torch.Size([2, 10, 2048])
Position IDs shape: torch.Size([2, 10])
Position IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Layer 0:
    当前 Q shape: torch.Size([2, 32, 10, 64])
    当前 K shape: torch.Size([2, 32, 10, 64])
    缓存后完整 K shape: torch.Size([2, 32, 10, 64])
    缓存后完整 V shape: torch.Size([2, 32, 10, 64])
  Layer 1:
    当前 Q shape: torch.Size([2, 32, 10, 64])
    当前 K shape: torch.Size([2, 32, 10, 64])
    缓存后完整 K shape: torch.Size([2, 32, 10, 64])
    缓存后完整 V shape: torch.Size([2, 32, 10, 64])
  Layer 2:
    当前 Q shape: torch.Size([2, 32, 10, 64])
    当前 K shape: torch.Size([2, 32, 10, 64])
    缓存后完整 K shape: torch.Size([2, 32, 10, 64])
    缓存后完整 V shape: torch.Size([2, 32, 10, 64])

Prefill完成后, KV cache中的序列长度: 10

============================================================
【Decode 阶段】逐个生成新token
============================================================

--- 生成第 1 个新token ---
新token embedding shape: torch.Size([2, 1, 2048])
当前position ID: 10
  Layer 0:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 11, 64])
    缓存后完整 V shape: torch.Size([2, 32, 11, 64])
  Layer 1:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 11, 64])
    缓存后完整 V shape: torch.Size([2, 32, 11, 64])
  Layer 2:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 11, 64])
    缓存后完整 V shape: torch.Size([2, 32, 11, 64])
当前KV cache总长度: 11

--- 生成第 2 个新token ---
新token embedding shape: torch.Size([2, 1, 2048])
当前position ID: 11
  Layer 0:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 12, 64])
    缓存后完整 V shape: torch.Size([2, 32, 12, 64])
  Layer 1:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 12, 64])
    缓存后完整 V shape: torch.Size([2, 32, 12, 64])
  Layer 2:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 12, 64])
    缓存后完整 V shape: torch.Size([2, 32, 12, 64])
当前KV cache总长度: 12

--- 生成第 3 个新token ---
新token embedding shape: torch.Size([2, 1, 2048])
当前position ID: 12
  Layer 0:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 13, 64])
    缓存后完整 V shape: torch.Size([2, 32, 13, 64])
  Layer 1:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 13, 64])
    缓存后完整 V shape: torch.Size([2, 32, 13, 64])
  Layer 2:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 13, 64])
    缓存后完整 V shape: torch.Size([2, 32, 13, 64])
当前KV cache总长度: 13

--- 生成第 4 个新token ---
新token embedding shape: torch.Size([2, 1, 2048])
当前position ID: 13
  Layer 0:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 14, 64])
    缓存后完整 V shape: torch.Size([2, 32, 14, 64])
  Layer 1:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 14, 64])
    缓存后完整 V shape: torch.Size([2, 32, 14, 64])
  Layer 2:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 14, 64])
    缓存后完整 V shape: torch.Size([2, 32, 14, 64])
当前KV cache总长度: 14

--- 生成第 5 个新token ---
新token embedding shape: torch.Size([2, 1, 2048])
当前position ID: 14
  Layer 0:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 15, 64])
    缓存后完整 V shape: torch.Size([2, 32, 15, 64])
  Layer 1:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 15, 64])
    缓存后完整 V shape: torch.Size([2, 32, 15, 64])
  Layer 2:
    当前 Q shape: torch.Size([2, 32, 1, 64])
    当前 K shape: torch.Size([2, 32, 1, 64])
    缓存后完整 K shape: torch.Size([2, 32, 15, 64])
    缓存后完整 V shape: torch.Size([2, 32, 15, 64])
当前KV cache总长度: 15

"""
