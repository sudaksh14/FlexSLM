import dataclasses
from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

import flex_modules as fm
from networks.config import FlexModelConfig, ModelConfig
from networks.flex_model import FlexModel
from typing import Optional


@dataclasses.dataclass
class FlexGPTConfig(FlexModelConfig):
    # GPT-2 tokeniser, Max-level dimensions must match
    # the target pretrained model exactly so that
    # HuggingFace weights can be loaded via utils.flexible_model_copy.
    vocab_size: int = 50257
    max_seq_length: int = 1024
    num_layers: int = 12
    hidden_dims: Iterable[int] = (384, 512, 768)
    num_heads: Iterable[int] = (6, 8, 12)
    mlp_dims: Iterable[int] = (1536, 2048, 3072)
    dropout: float = 0.1
    tie_embeddings: bool = True
    pretrained_hf_model: Optional[str] = None   # e.g. "gpt2"

    def make_model(self) -> 'FlexGPT':
        return FlexGPT(self)

    def no_prebuilt(self) -> 'FlexGPTConfig':
        return self

    # def create_base_config(self, level) -> ModelConfig:

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1


class MLPBlock(nn.Sequential):
    def __init__(self, hidden_dim: Iterable[int], mlp_dim: Iterable[int], dropout: float):
        super().__init__(
            fm.Linear(hidden_dim, mlp_dim, bias=True),
            # nn.GELU(approximate='tanh'),
            nn.GELU(),
            nn.Dropout(dropout),
            fm.Linear(mlp_dim, hidden_dim, bias=True),
            nn.Dropout(dropout),
        )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: Iterable[int],
        num_heads: Iterable[int],
        mlp_dim: Iterable[int],
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.ln_1 = fm.LayerNorm(hidden_dim, eps=1e-5)
        self.attn = fm.SelfAttention(hidden_dim, num_heads, dropout=attention_dropout, is_causal=True)
        self.ln_2 = fm.LayerNorm(hidden_dim, eps=1e-5)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class TiedLMHead(nn.Module):
    """
    LM head that shares weights with FlexEmbedding (weight tying).
    """

    def __init__(self, embedding: fm.Embedding):
        super().__init__()
        self._embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_dim = self._embedding.hidden_dims[self._embedding.level]
        w = self._embedding.weight[:, :hidden_dim]  # [vocab_size, hidden_dim]
        return F.linear(x, w)                       # [B, S, vocab_size]


class FlexGPT(FlexModel):
    """
    Flexible decoder-only language model.
    Width (hidden_dim, heads, mlp_dim) varies per level.
    """

    def __init__(self, config: FlexGPTConfig):
        super().__init__(config)

        hidden_dims = list(config.hidden_dims)
        num_heads   = list(config.num_heads)
        mlp_dims    = list(config.mlp_dims)

        self.hidden_dims = hidden_dims

        # Input
        self.token_embedding = fm.Embedding(config.vocab_size, hidden_dims)
        self.pos_embedding   = fm.PosEmbeddingLayer(config.max_seq_length, hidden_dims)
        self.emb_dropout     = nn.Dropout(config.dropout)

        # Transformer stack
        self.blocks = nn.Sequential(OrderedDict(
            {f"block_{i}": DecoderBlock(hidden_dims, num_heads, mlp_dims, config.dropout, config.dropout)
             for i in range(config.num_layers)}
        ))

        # Output
        self.ln_f = fm.LayerNorm(hidden_dims, eps=1e-5)

        if config.tie_embeddings:
            self.lm_head = TiedLMHead(self.token_embedding)
        else:
            self.lm_head = fm.LinearSelect(
                hidden_dims, [config.vocab_size] * len(hidden_dims), bias=False)

        self.set_level_use(self.max_level())
        self.level = self.max_level()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, S]
        x = self.token_embedding(input_ids)   # [B, S, hidden_dim]
        x = self.pos_embedding(x)             # [B, S, hidden_dim]
        x = self.emb_dropout(x)
        x = self.blocks(x)                    # [B, S, hidden_dim]
        x = self.ln_f(x)                      # [B, S, hidden_dim]
        return self.lm_head(x)                # [B, S, vocab_size]

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


torch.serialization.add_safe_globals([FlexGPTConfig])
