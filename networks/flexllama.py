import dataclasses
from collections import OrderedDict
from typing import Iterable, Optional

import torch
from torch import nn

import flex_modules as fm
from networks.config import FlexModelConfig
from networks.flex_model import FlexModel


@dataclasses.dataclass
class FlexLLaMAConfig(FlexModelConfig):
    vocab_size: int = 32000
    max_seq_length: int = 1024
    num_layers: int = 12
    hidden_dims: Iterable[int] = (256, 512, 768)
    num_heads: Iterable[int] = (4, 8, 12)
    num_kv_heads: Iterable[int] = (4, 8, 12)
    intermediate_dims: Iterable[int] = (1024, 2048, 3072)
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = False
    qkv_bias: bool = False  # True for Qwen2-family checkpoints; LLaMA family has no attention bias
    pretrained_hf_model: Optional[str] = None

    def make_model(self) -> 'FlexLLaMA':
        return FlexLLaMA(self)

    def no_prebuilt(self) -> 'FlexLLaMAConfig':
        return dataclasses.replace(self, pretrained_hf_model=None)

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1


torch.serialization.add_safe_globals([FlexLLaMAConfig])


class LLaMAMLPBlock(nn.Module):
    """
    SwiGLU feed-forward block.
    """

    def __init__(self, hidden_dims: Iterable[int], intermediate_dims: Iterable[int]):
        super().__init__()
        self.gate_proj = fm.Linear(hidden_dims, intermediate_dims, bias=False)
        self.up_proj   = fm.Linear(hidden_dims, intermediate_dims, bias=False)
        self.down_proj = fm.Linear(intermediate_dims, hidden_dims, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMADecoderBlock(nn.Module):
    """
    Single LLaMA transformer block.
    Pre-norm with RMSNorm, RoPE causal attention, SwiGLU MLP, no dropout.
    """

    def __init__(
        self,
        hidden_dims: Iterable[int],
        num_heads: Iterable[int],
        intermediate_dims: Iterable[int],
        rope_theta: float,
        rms_norm_eps: float,
        max_seq_len: int,
        num_kv_heads: Iterable[int] = None,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.input_layernorm          = fm.RMSNorm(hidden_dims, eps=rms_norm_eps)
        self.self_attn                = fm.RoPEAttention(hidden_dims, num_heads,
                                                         num_kv_heads=num_kv_heads,
                                                         rope_theta=rope_theta,
                                                         max_seq_len=max_seq_len,
                                                         qkv_bias=qkv_bias)
        self.post_attention_layernorm = fm.RMSNorm(hidden_dims, eps=rms_norm_eps)
        self.mlp                      = LLaMAMLPBlock(hidden_dims, intermediate_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class FlexLLaMA(FlexModel):
    """
    Flexible decoder-only language model built on the LLaMA architecture.
    Width (hidden_dim, heads, intermediate_dim) varies per level.
    """

    def __init__(self, config: FlexLLaMAConfig):
        super().__init__(config)

        hidden_dims       = list(config.hidden_dims)
        num_heads         = list(config.num_heads)
        num_kv_heads      = list(config.num_kv_heads)
        intermediate_dims = list(config.intermediate_dims)
        n_levels          = len(hidden_dims)

        self.hidden_dims = hidden_dims

        self.embed_tokens = fm.Embedding(config.vocab_size, hidden_dims)

        self.layers = nn.Sequential(OrderedDict(
            {f"layer_{i}": LLaMADecoderBlock(
                hidden_dims, num_heads, intermediate_dims,
                config.rope_theta, config.rms_norm_eps, config.max_seq_length,
                num_kv_heads=num_kv_heads, qkv_bias=config.qkv_bias,
            ) for i in range(config.num_layers)}
        ))

        self.norm = fm.RMSNorm(hidden_dims, eps=config.rms_norm_eps)

        if config.tie_embeddings:
            self.lm_head = _TiedLMHead(self.embed_tokens)
        else:
            self.lm_head = fm.LinearSelect(
                hidden_dims, [config.vocab_size] * n_levels, bias=False)

        self.set_level_use(self.max_level())
        self.level = self.max_level()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)   # [B, S, hidden_dim]
        x = self.layers(x)                 # [B, S, hidden_dim]
        x = self.norm(x)                   # [B, S, hidden_dim]
        return self.lm_head(x)             # [B, S, vocab_size]

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @torch.no_grad()
    def export_level_delta(self):
        delta_down, delta_up = super().export_level_delta()
        return fm.DownDelta(
            (self.hidden_dims[self.level], delta_down)
        ), fm.UpDelta(
            (self.hidden_dims[self.level], delta_up))

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: fm.DownDelta) -> None:
        hidden_dim, module_deltas = level_delta.delta
        FlexModel.apply_level_delta_down(model, module_deltas)
        model.hidden_dim = hidden_dim

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: fm.UpDelta) -> None:
        hidden_dim, module_deltas = level_delta.delta
        FlexModel.apply_level_delta_up(model, module_deltas)
        model.hidden_dim = hidden_dim


class _TiedLMHead(nn.Module):
    """LM head sharing weights with fm.Embedding. Currently not used."""

    def __init__(self, embedding: fm.Embedding):
        super().__init__()
        self._embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self._embedding.hidden_dims[self._embedding.level]
        w = self._embedding.weight[:, :d]
        return torch.nn.functional.linear(x, w)
