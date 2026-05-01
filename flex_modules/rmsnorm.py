from typing import Iterable

import torch
from torch import nn

from flex_modules.module import Module, DownDelta, UpDelta


class _RMSNormBase(nn.Module):
    """Plain RMSNorm. Base_type for fm.RMSNorm."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class RMSNorm(Module):
    """
    Flexible RMSNorm. Stores a single weight vector at max hidden_dim
    and slices to hidden_dims[level] on each forward pass.
    """

    def __init__(self, hidden_dims: Iterable[int], eps: float = 1e-5):
        super().__init__()
        hidden_dims = list(hidden_dims)
        assert len(hidden_dims) > 0
        assert max(hidden_dims) == hidden_dims[-1], "hidden_dims must be non-decreasing"
        self.hidden_dims = hidden_dims
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dims[-1]))
        self.level = self.max_level()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.hidden_dims[self.level]
        w = self.weight[:d]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * w

    def set_level_use(self, level: int) -> None:
        assert 0 <= level <= self.max_level()
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[_RMSNormBase]:
        return _RMSNormBase

    def copy_to_base(self, dest: _RMSNormBase) -> None:
        d = self.hidden_dims[self.level]
        dest.weight.data = self.weight.data[:d].clone()

    def load_from_base(self, src: _RMSNormBase) -> None:
        d = self.hidden_dims[self.level]
        self.weight.data[:d] = src.weight.data

    def _make_reg_layer(self) -> _RMSNormBase:
        return _RMSNormBase(self.hidden_dims[self.level], self.eps)

    def export_level_delta(self) -> tuple[DownDelta, UpDelta]:
        new_elems = self.weight.data[
            self.hidden_dims[self.level - 1]:self.hidden_dims[self.level]
        ].clone()
        return DownDelta(self.hidden_dims[self.level]), UpDelta(new_elems)

    @staticmethod
    def apply_level_delta_down(model: _RMSNormBase, level_delta: DownDelta) -> None:
        hidden_dim = level_delta.delta
        model.weight.data = model.weight.data[:hidden_dim].contiguous()
        model.hidden_dim = hidden_dim

    @staticmethod
    def apply_level_delta_up(model: _RMSNormBase, level_delta: UpDelta) -> None:
        new_elems = level_delta.delta.to(model.weight.data)
        model.weight.data = torch.cat([model.weight.data, new_elems]).contiguous()
        model.hidden_dim = model.weight.data.shape[0]
