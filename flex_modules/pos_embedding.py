from typing import Any, Iterable

from torch import nn
import torch

from flex_modules.module import Module, UpDelta, DownDelta
import networks.modules as vmod


class PosEmbeddingLayer(Module):
    def __init__(self, seq_length, hidden_dim: Iterable[int]):
        super().__init__()

        def is_positive(x): return x > 0
        assert (is_positive(seq_length))
        assert (all(map(is_positive, hidden_dim)))
        assert (max(hidden_dim) == hidden_dim[-1])

        self.seq_length = seq_length
        self.hidden_dims = hidden_dim
        self.embedding = nn.Parameter(torch.empty(
            1, seq_length, hidden_dim[-1]).normal_(std=0.02))
        self.level = self.max_level()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.embedding[:, :seq_len, :self.hidden_dims[self.level]]

    def set_level_use(self, level: int) -> None:
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[nn.Module]:
        return vmod.PosEmbeddingLayer

    @torch.no_grad()
    def copy_to_base(self, dest: vmod.PosEmbeddingLayer) -> None:
        dest.embedding.data = self.embedding.data[
            :, :, :self.hidden_dims[self.level]]

    @torch.no_grad()
    def load_from_base(self, src: vmod.PosEmbeddingLayer) -> None:
        self.embedding.data[
            :, :, :self.hidden_dims[self.level]] = src.embedding.data

    def _make_reg_layer(self):
        return vmod.PosEmbeddingLayer(
            self.seq_length, self.hidden_dims[self.level])

    @torch.no_grad()
    def export_level_delta(self) -> tuple[DownDelta[int], UpDelta[torch.Tensor]]:
        return (
            DownDelta(self.hidden_dims[self.level]),
            UpDelta(self.embedding[
                :, :,
                self.hidden_dims[self.level - 1]:self.hidden_dims[self.level]
            ])
        )

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: vmod.PosEmbeddingLayer, level_delta: DownDelta[int]) -> None:
        model.embedding.data = model.embedding.data[:, :, :level_delta.delta].to(
            model.embedding.data)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: vmod.PosEmbeddingLayer, level_delta: UpDelta[torch.Tensor]) -> None:
        model.embedding.data = torch.cat(
            [model.embedding.data, level_delta.delta.to(model.embedding.data)], dim=2)
