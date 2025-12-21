from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
import torch
from torch import nn
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from torch_interface_test.torch_lib import create_tensor
from utils import DynamicCache
from utils import _prepare_decoder_attention_mask


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
    # print(f"q shape: {q.shape}, k shape: {k.shape}")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:  # [B, N2, S2, D]
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 广播 [B, N2, 1, S2, D] -> [B, N2, G, S2, D]
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)  # [B, N2, G, S2, D]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)  # [B, N2*G, S2, D]


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, N1, S1, D]
    key: torch.Tensor,  # [B, N2, S2, D]
    value: torch.Tensor,  # [B, N2, S2, D]
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Group attention implementation
    key_states = repeat_kv(key, module.num_key_value_groups)  # [B, N2 * G, S2, D]
    value_states = repeat_kv(value, module.num_key_value_groups)  # [B, N2 * G, S2, D]

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # [B, N1, S1, S2]
    if attention_mask is not None:
        # 切片操作：截取到 key_states 的序列长度
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]  # [B, N1, S1, S2]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  # [B, N1, S1, S2]
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)  # [B, N1, S1, D]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S1, N1, D]

    return attn_output, attn_weights


class LlamaAttention(nn.Module):

    def __init__(self, config: TinyLlama_Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        # self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary_fn = apply_rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,S,H]
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # past_key_values: Cache = None,
        past_key_values: DynamicCache = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]  # [B,S1]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [B,S1,-1,D]

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,N1,S1,D]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,N2,S1,D]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,N2,S1,D]

        cos, sin = position_embeddings  # [B,S,D]
        # 在 N 维度扩展
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # [B,N,S,D]

        # 拼接 key_states, value_states 用于生成

        # KV cache logic (for generation)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)  # [B,N,S2,D]

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )  # [B, S1, N1, D]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # [B, S1, H]
        attn_output = self.o_proj(attn_output)  # [B, S1, H]
        return attn_output, attn_weights


def test_llama_attention():
    B = 3
    N1 = 32
    N2 = 4
    G = 8
    S = 6
    D = 64
    H = 2048
    num_layers = 2
    past_key_values_length = 0
    config = TinyLlama_Config()
    hidden_states = create_tensor((B, S, config.hidden_size), dtype=torch.float16, ndim=3, device="cuda")  # [B,S,H]
    # padding mask
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],  # 第1句：5个真实token + 1个padding
            [1, 1, 0, 0, 0, 0],  # 第2句：2个真实token + 4个padding
            [1, 1, 1, 0, 0, 0],  # 第3句：3个真实token + 3个padding
        ]
    )
    layers = [LlamaAttention(config, layer_idx=i).to(device="cuda", dtype=torch.float16) for i in range(num_layers)]
    for layer in layers:
        layer.eval()
    sin_cache = create_tensor((1, config.max_position_embeddings, D), dtype=torch.float16, ndim=3, device="cuda")
    cos_cache = create_tensor((1, config.max_position_embeddings, D), dtype=torch.float16, ndim=3, device="cuda")
    past_key_values = DynamicCache()

    # prefill [B, 1, S, S + past_len]
    attention_mask = _prepare_decoder_attention_mask(attention_mask, hidden_states.shape[:-1], hidden_states, past_key_values_length)
    position_embeddings = (cos_cache[:, :S, :], sin_cache[:, :S, :])  # [B,S,D]

    for layer_idx, llama_attn in enumerate(layers):
        attn_output, attn_weights = llama_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        print(f"Layer {layer_idx} output shape: {attn_output.shape}, attn_weights shape: {attn_weights.shape}")
        print(f"kv cache length {past_key_values.get_seq_length()}")


if __name__ == "__main__":
    test_llama_attention()

"""
Output Log:
Layer 0 output shape: torch.Size([3, 6, 2048]), attn_weights shape: torch.Size([3, 32, 6, 6])
kv cache length 6
Layer 1 output shape: torch.Size([3, 6, 2048]), attn_weights shape: torch.Size([3, 32, 6, 6])
kv cache length 6
"""
