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


def _interp_rows(rows: torch.Tensor, widths: list, d: int) -> torch.Tensor:
    """rows[i] belongs to width widths[i] (increasing). The row for width d: exact at a
    trained width, linear in d between two, clamped outside [widths[0], widths[-1]]."""
    if d <= widths[0]:
        return rows[0]
    if d >= widths[-1]:
        return rows[-1]
    for i in range(len(widths) - 1):
        lo, hi = widths[i], widths[i + 1]
        if d == lo:
            return rows[i]
        if lo < d < hi:
            t = (d - lo) / (hi - lo)
            return rows[i] * (1 - t) + rows[i + 1] * t
    return rows[-1]


class RMSNorm(Module):
    """
    Flexible RMSNorm. Stores a single weight vector at max hidden_dim
    and slices to hidden_dims[level] on each forward pass.
    """

    def __init__(self, hidden_dims: Iterable[int], eps: float = 1e-5, level_norms: bool = False):
        super().__init__()
        hidden_dims = list(hidden_dims)
        assert len(hidden_dims) > 0
        assert max(hidden_dims) == hidden_dims[-1], "hidden_dims must be non-decreasing"
        self.hidden_dims = hidden_dims
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dims[-1]))
        self.level = self.max_level()
        self.dim_override = None  # any prefix width (FlexLLaMA.set_widths); cleared by set_level_use
        # Per-width norm weights (switchable norms of slimmable networks): level l uses
        # weight + level_delta[l]. Zero-initialised, so training starts from the shared
        # weight. A width between two trained levels interpolates their deltas linearly
        # in the width; below the smallest level it uses the smallest level's delta.
        self.level_norms = level_norms
        if level_norms:
            assert all(a < b for a, b in zip(hidden_dims, hidden_dims[1:])), "level_norms needs increasing widths"
            self.level_delta = nn.Parameter(torch.zeros(len(hidden_dims), hidden_dims[-1]))

    def current_dim(self) -> int:
        return self.dim_override or self.hidden_dims[self.level]

    def effective_weight(self) -> torch.Tensor:
        d = self.current_dim()
        if not self.level_norms:
            return self.weight[:d]
        return self.weight[:d] + _interp_rows(self.level_delta, self.hidden_dims, d)[:d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * w

    def set_level_use(self, level: int) -> None:
        assert 0 <= level <= self.max_level()
        self.level = level
        self.dim_override = None

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[_RMSNormBase]:
        return _RMSNormBase

    def copy_to_base(self, dest: _RMSNormBase) -> None:
        dest.weight.data = self.effective_weight().detach().clone()

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
