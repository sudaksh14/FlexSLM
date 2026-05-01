from typing import Iterable

from torch import nn
import torch

from flex_modules.module import Module, DownDelta, UpDelta


class Embedding(Module):
    """
    Flexible token embedding table. Stores weights at max hidden_dim and slices
    columns per level.
    """

    def __init__(self, vocab_size: int, hidden_dims: Iterable[int], **kwargs):
        super().__init__()

        hidden_dims = list(hidden_dims)
        assert len(hidden_dims) > 0
        assert max(hidden_dims) == hidden_dims[-1], "hidden_dims must be increasing!!"

        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self._kwargs = kwargs

        self.level = self.max_level()
        self.embedding = nn.Embedding(vocab_size, hidden_dims[-1], **kwargs)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)                          # [B, S, max_hidden_dim]
        return out[..., :self.hidden_dims[self.level]]   # [B, S, hidden_dim[level]]

    def set_level_use(self, level: int) -> None:
        assert 0 <= level <= self.max_level()
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[nn.Embedding]:
        return nn.Embedding

    def copy_to_base(self, dest: nn.Embedding) -> None:
        dest.weight.data = self.embedding.weight.data[:, :self.hidden_dims[self.level]]

    def load_from_base(self, src: nn.Embedding) -> None:
        self.embedding.weight.data[:, :self.hidden_dims[self.level]] = src.weight.data

    def _make_reg_layer(self) -> nn.Embedding:
        return nn.Embedding(self.vocab_size, self.hidden_dims[self.level], **self._kwargs)

    def export_level_delta(self) -> tuple[DownDelta, UpDelta]:
        new_cols = self.embedding.weight.data[
            :, self.hidden_dims[self.level - 1]:self.hidden_dims[self.level]
        ].clone()
        return DownDelta(self.hidden_dims[self.level]), UpDelta(new_cols)

    @staticmethod
    def apply_level_delta_down(model: nn.Embedding, level_delta: DownDelta) -> None:
        hidden_dim = level_delta.delta
        model.weight.data = model.weight.data[:, :hidden_dim].contiguous()
        model.embedding_dim = hidden_dim

    @staticmethod
    def apply_level_delta_up(model: nn.Embedding, level_delta: UpDelta) -> None:
        new_cols = level_delta.delta.to(model.weight.data)
        model.weight.data = torch.cat([model.weight.data, new_cols], dim=1).contiguous()
        model.embedding_dim = model.weight.data.shape[1]
