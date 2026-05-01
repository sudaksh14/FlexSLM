"""
rope operation largely adapted from https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from flex_modules.linear import Linear


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:   [B, H, S, head_dim]
    # cos/sin: [S, head_dim], broadcast over B and H dims
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class RoPEAttention(nn.Module):
    """
    Multi-head causal self-attention with Rotary Position Embeddings (RoPE).
    """

    def __init__(
        self,
        hidden_dims: Iterable[int],
        num_heads: Iterable[int],
        rope_theta: float = 10000.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        num_heads = list(num_heads)
        assert len(hidden_dims) == len(num_heads)
        assert all(h % n == 0 for h, n in zip(hidden_dims, num_heads))

        self.hidden_dims = hidden_dims
        self.num_heads_list = num_heads

        # Separate projections, no bias (LLaMA convention)
        self.q_proj = Linear(hidden_dims, hidden_dims, bias=False)
        self.k_proj = Linear(hidden_dims, hidden_dims, bias=False)
        self.v_proj = Linear(hidden_dims, hidden_dims, bias=False)
        self.o_proj = Linear(hidden_dims, hidden_dims, bias=False)

        # Precompute RoPE cos/sin per level.
        for level in range(len(hidden_dims)):
            head_dim = hidden_dims[level] // num_heads[level]
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )
            t = torch.arange(max_seq_len).float()
            freqs = torch.outer(t, inv_freq)          # [max_seq_len, head_dim // 2]
            emb = torch.cat([freqs, freqs], dim=-1)   # [max_seq_len, head_dim]
            self.register_buffer(f'cos_{level}', emb.cos())
            self.register_buffer(f'sin_{level}', emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, hidden_dim]
        B, S, _ = x.shape
        level = self.q_proj.current_level()
        num_heads = self.num_heads_list[level]
        hidden_dim = self.hidden_dims[level]
        head_dim = hidden_dim // num_heads

        q = self.q_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)

        cos = getattr(self, f'cos_{level}')[:S]   # [S, head_dim]
        sin = getattr(self, f'sin_{level}')[:S]   # [S, head_dim]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, hidden_dim)
        return self.o_proj(out)
