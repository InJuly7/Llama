from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable


@dataclass
class ModelConfig:
    architectures: str = "LlamaForCausalLM"
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    max_sequence_length = 4096
    model_type: str = "llama"
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    head_dim: int = 128
    num_key_value_heads: int = 32
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    rope_scaling = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "float16"
    transformers_version: str = "4.31.0.dev0"
    use_cache: bool = True
    vocab_size: int = 32000
    device = "cuda"
    rope_parameters: dict = None
    rope_type: str = "default"
    mlp_bias: bool = False
    pad_token_id: int = 0


# Llama 7B
@dataclass
class Llama_7B_Config(ModelConfig):
    max_sequence_length = 2048
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    transformers_version: str = "4.28.0.dev0"


# Llama2 7B
@dataclass
class Llama2_7B_Config(ModelConfig):
    rope_parameters: dict = field(default_factory=lambda: {"rope_type": "default", "rope_theta": 10000.0})


@dataclass
class TinyLlama_Config(ModelConfig):
    hidden_size: int = 2048
    intermediate_size: int = 5632
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 22
    num_key_value_heads: int = 4
    rope_theta = 10000.0
    torch_dtype = "bfloat16"
    transformers_version = "4.35.0"
    attention_bias = False
    head_dim: int = 64
    rope_parameters: dict = field(default_factory=lambda: {"rope_type": "default", "rope_theta": 10000.0})
    logits_to_keep = 1
    do_sample = False
