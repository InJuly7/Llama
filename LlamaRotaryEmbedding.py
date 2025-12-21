from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
import torch
from torch import nn
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from torch_interface_test.torch_lib import create_tensor


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Llama2_7B_Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        # if self.rope_type != "default":
        #     rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        # 不会被保存到 checkpoint 中
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 备份
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Llama2_7B_Config] = None,
        device: Optional["torch.device"] = None,
        S: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:

        base = config.rope_parameters["rope_theta"]
        D = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        # 偶数索引, 对应RoPE中的维度分组 (每对维度共享一个频率)
        # \theta = (10000)^{-2i/d} i = 0,1,2,...,d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / D))  # [D // 2]
        return inv_freq, attention_factor

    @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # 扩展 inv_freq 到与输入兼容的形状
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)  # [B, D // 2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [B,1,S]
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # [B, S, D // 2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
            cos = emb.cos() * self.attention_scaling  # [B, S, D]
            sin = emb.sin() * self.attention_scaling  # [B, S, D]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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


def test_llama_rope():
    config = TinyLlama_Config()
    rope = LlamaRotaryEmbedding(config=config, device=config.device)
    if config.torch_dtype == "float16":
        dtype = torch.float16
    elif config.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    rope.eval()
    q_seq_max = 3
    k_seq_max = 3
    # [1, max_position_embeddings]
    position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long, device=config.device).unsqueeze(0)  # [B, S]
    q = create_tensor((1, q_seq_max, config.num_attention_heads, config.head_dim), dtype=dtype, ndim=4, device=config.device)
    k = create_tensor((1, k_seq_max, config.num_attention_heads, config.head_dim), dtype=dtype, ndim=4, device=config.device)
    q = q.transpose(1, 2)  # [B,N,S,D]
    k = k.transpose(1, 2)  # [B,N,S,D]
    cos, sin = rope.forward(q, position_ids)  # [B, max_position_embeddings, D]
    # 在 N 维度扩展
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos[:, :k_seq_max, :], sin[:, :k_seq_max, :], unsqueeze_dim=1)  # [B, N, S, D]
    print(f"q_embed shape: {q_embed.shape}")
    print(f"k_embed shape: {k_embed.shape}")


# 生成旋转矩阵
def precompute_freqs_cis(D: int, S: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, D, 2)[: (D // 2)].float() / D))  # [D // 2]
    # 生成 token 序列索引 t = [0, 1,..., S-1]
    t = torch.arange(S, device=freqs.device)  # [S]
    # 计算m * \theta
    freqs = torch.outer(t, freqs).float()  # [S, D // 2]

    # 计算结果是个复数向量
    # 假设 freqs = [\theta_0, \theta_1, ..., \theta_{D-1//2}], r = [1, 1, ..., 1]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [S, D // 2]
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # [B, S, N, D]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [B, S, N, D]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)  # [B, S, N, D // 2, 2]
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)  # [B, S, N, D // 2, 2]

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)  # [B, S, N, D // 2]
    xk_ = torch.view_as_complex(xk_)  # [B, S, N, D // 2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # [1, S, 1, D // 2]
    # 应用旋转操作，然后将结果转回实数域
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)  # [B, S, H]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)  # [B, S, H]
    return xq_out.type_as(xq), xk_out.type_as(xk)


def test_rope():
    B = 1
    S = 3
    N = 32
    D = 128
    max_seq_len = 2048
    # 乘2 是为了避免推理时候动态计算频率, 即使超出最大窗口是输出质量下降
    freqs_cis = precompute_freqs_cis(D, max_seq_len * 2)  # [max_seq_len * 2, D // 2]

    # B,S,N,D
    xq = torch.randn(B, S, N, D)  # [B, S, N, D]
    xk = torch.randn(B, S, N, D)  # [B, S, N, D]

    freqs_cis_reshape = freqs_cis[:S, :]  # [S, D // 2]
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis_reshape)  # [B, S, H]
    print(f"xq_out shape: {xq_out.shape}")
    print(f"xk_out shape: {xk_out.shape}")


if __name__ == "__main__":
    test_rope()
    test_llama_rope()
