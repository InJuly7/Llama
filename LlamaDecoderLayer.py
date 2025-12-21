import torch
from torch import nn
from typing import Optional, Tuple, List
from dataclasses import dataclass

from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
from LlamaMLP import LlamaMLP
from LlamaAttention import LlamaAttention
from LlamaRMSNorm import LlamaRMSNorm
from LlamaRotaryEmbedding import LlamaRotaryEmbedding

from utils import DynamicCache
from utils import _prepare_decoder_attention_mask


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: TinyLlama_Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def test_llama_decoder_layer():
    config = TinyLlama_Config()
    layer = LlamaDecoderLayer(config, layer_idx=0).to(device=config.device)
    layer.eval()

    print("=" * 50)
    print("PREFILL 阶段")
    print("=" * 50)

    # prefill 阶段
    batch_size = 2
    seq_length = 5
    hidden_size = config.hidden_size
    past_key_values_length = 0

    hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(dtype=torch.float16, device="cuda")
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0).to(dtype=torch.long, device="cuda")  # 改为 long
    print(f"hidden_states shape: {hidden_states.shape}")
    print(f"position_ids shape: {position_ids.shape}")
    print(f"position_ids: {position_ids}")

    attention_mask = None
    attention_mask = _prepare_decoder_attention_mask(
        attention_mask,
        hidden_states.shape[:-1],
        hidden_states,
        past_key_values_length=past_key_values_length,
    )
    print(f"Attention Mask shape: {attention_mask.shape}")
    print(f"Attention Mask:\n{attention_mask}")

    # 创建 KV cache（重要：在 prefill 和 decode 之间共享）
    past_key_values = DynamicCache()
    use_cache = True
    cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device="cuda")
    print(f"cache_position: {cache_position}")

    rope = LlamaRotaryEmbedding(config).to(device=config.device)
    position_embedding = rope(hidden_states, position_ids)
    print(f"Position Embedding Sin shape: {position_embedding[0].shape}")

    output = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embedding,
    )
    print(f"Output shape: {output.shape}")
    print(f"KV Cache keys shape: {past_key_values.key_cache[0].shape}")
    print(f"KV Cache values shape: {past_key_values.value_cache[0].shape}")

    print("\n" + "=" * 50)
    print("DECODE 阶段")
    print("=" * 50)

    # Decode 阶段 - 生成下一个 token
    batch_size = 2
    hidden_size = config.hidden_size
    past_key_values_length = seq_length  # 之前已经缓存了 5 个 token
    new_token_length = 1  # decode 阶段每次生成 1 个 token

    hidden_states = torch.randn(batch_size, new_token_length, hidden_size).to(dtype=torch.float16, device="cuda")
    # position_ids 应该是当前 token 的位置（即第 5 个位置，索引为 5）
    position_ids = torch.tensor([[past_key_values_length]], dtype=torch.long, device="cuda")  # [[5]]
    print(f"hidden_states shape: {hidden_states.shape}")
    print(f"position_ids shape: {position_ids.shape}")
    print(f"position_ids: {position_ids}")

    # Decode 阶段的 attention mask: 当前 token 可以 attend 到所有之前的 token
    attention_mask = None
    attention_mask = _prepare_decoder_attention_mask(
        attention_mask,
        (batch_size, new_token_length),  # 当前只有 1 个 token
        hidden_states,
        past_key_values_length=past_key_values_length,  # 之前已经有 5 个 token
    )

    # 重要：使用之前的 cache，而不是创建新的
    use_cache = True
    cache_position = torch.tensor([past_key_values_length], device="cuda")  # [5]
    print(f"cache_position: {cache_position}")

    position_embedding = rope(hidden_states, position_ids)
    print(f"Position Embedding Sin shape: {position_embedding[0].shape}")
    print(f"Position Embedding Sin:\n{position_embedding[0]}")

    output = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,  # 传入 prefill 阶段的 cache
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embedding,
    )
    print(f"Output shape: {output.shape}")
    print(f"KV Cache keys shape: {past_key_values.key_cache[0].shape}")  # 应该是 [B, num_heads, 6, head_dim]
    print(f"KV Cache values shape: {past_key_values.value_cache[0].shape}")

    print("\n" + "=" * 50)
    print("再次 DECODE(生成第 7 个 token)")
    print("=" * 50)

    # 可以继续 decode
    past_key_values_length = 6
    hidden_states = torch.randn(batch_size, 1, hidden_size).to(dtype=torch.float16, device="cuda")
    position_ids = torch.tensor([[past_key_values_length]], dtype=torch.long, device="cuda")  # [[6]]

    attention_mask = _prepare_decoder_attention_mask(
        None,
        (batch_size, 1),
        hidden_states,
        past_key_values_length=past_key_values_length,
    )

    cache_position = torch.tensor([past_key_values_length], device="cuda")
    position_embedding = rope(hidden_states, position_ids)

    output = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,  # 继续使用之前的 cache
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embedding,
    )
    print(f"Output shape: {output.shape}")
    print(f"KV Cache keys shape: {past_key_values.key_cache[0].shape}")  # 应该是 [B, num_heads, 7, head_dim]
    print(f"KV Cache values shape: {past_key_values.value_cache[0].shape}")


if __name__ == "__main__":
    test_llama_decoder_layer()

"""
Output Log:
(/home/song/conda_env/Llama) song@Orin-Nx:~/program/Llama$ python3 LlamaDecoderLayer.py 
==================================================
PREFILL 阶段
==================================================
hidden_states shape: torch.Size([2, 5, 2048])
position_ids shape: torch.Size([1, 5])
position_ids: tensor([[0, 1, 2, 3, 4]], device='cuda:0')
Attention Mask shape: torch.Size([2, 1, 5, 5])
Attention Mask:
tensor([[[[     0., -65504., -65504., -65504., -65504.],
          [     0.,      0., -65504., -65504., -65504.],
          [     0.,      0.,      0., -65504., -65504.],
          [     0.,      0.,      0.,      0., -65504.],
          [     0.,      0.,      0.,      0.,      0.]]],


        [[[     0., -65504., -65504., -65504., -65504.],
          [     0.,      0., -65504., -65504., -65504.],
          [     0.,      0.,      0., -65504., -65504.],
          [     0.,      0.,      0.,      0., -65504.],
          [     0.,      0.,      0.,      0.,      0.]]]], device='cuda:0',
       dtype=torch.float16)
cache_position: tensor([0, 1, 2, 3, 4], device='cuda:0')
Position Embedding Sin shape: torch.Size([1, 5, 64])
Output shape: torch.Size([2, 5, 2048])
KV Cache keys shape: torch.Size([2, 4, 5, 64])
KV Cache values shape: torch.Size([2, 4, 5, 64])

==================================================
DECODE 阶段
==================================================
hidden_states shape: torch.Size([2, 1, 2048])
position_ids shape: torch.Size([1, 1])
position_ids: tensor([[5]], device='cuda:0')
cache_position: tensor([5], device='cuda:0')
Position Embedding Sin shape: torch.Size([1, 1, 64])
Position Embedding Sin:
tensor([[[ 0.2837, -0.8208, -0.9463, -0.5122, -0.0103,  0.3757,  0.6299,
           0.7856,  0.8774,  0.9307,  0.9609,  0.9780,  0.9873,  0.9932,
           0.9961,  0.9976,  0.9985,  0.9995,  0.9995,  1.0000,  1.0000,
           1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
           1.0000,  1.0000,  1.0000,  1.0000,  0.2837, -0.8208, -0.9463,
          -0.5122, -0.0103,  0.3757,  0.6299,  0.7856,  0.8774,  0.9307,
           0.9609,  0.9780,  0.9873,  0.9932,  0.9961,  0.9976,  0.9985,
           0.9995,  0.9995,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
           1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
           1.0000]]], device='cuda:0', dtype=torch.float16)
Output shape: torch.Size([2, 1, 2048])
KV Cache keys shape: torch.Size([2, 4, 6, 64])
KV Cache values shape: torch.Size([2, 4, 6, 64])

==================================================
再次 DECODE(生成第 7 个 token)
==================================================
Output shape: torch.Size([2, 1, 2048])
KV Cache keys shape: torch.Size([2, 4, 7, 64])
KV Cache values shape: torch.Size([2, 4, 7, 64])
"""
