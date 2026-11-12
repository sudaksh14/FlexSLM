from typing import Iterable, Any

from torch import nn
import torch.nn.functional as F
import torch

from flex_modules.module import Module, DownDelta, UpDelta


class Linear(Module):
    def __init__(self, in_sizes: Iterable[int], out_sizes: Iterable[int], *args, **kwargs):
        super().__init__()

        def is_positive(x): return x > 0
        assert (len(in_sizes) > 0)
        assert (len(in_sizes) == len(out_sizes))
        assert (all(map(is_positive, in_sizes)))
        assert (all(map(is_positive, out_sizes)))
        assert (max(in_sizes) == in_sizes[-1])
        assert (max(out_sizes) == out_sizes[-1])

        self.in_sizes = in_sizes
        self.out_sizes = out_sizes

        self._args = args
        self._kwargs = kwargs

        self.max_in_size = self.in_sizes[-1]
        self.max_out_size = self.out_sizes[-1]

        self.level = self.max_level()
        # (in, out) widths that replace in_sizes/out_sizes[level] when set: any prefix
        # width, not only a trained level (FlexLLaMA.set_widths). set_level_use clears it.
        self.dims_override = None
        self.linear = nn.Linear(
            self.max_in_size, self.max_out_size, *args, **kwargs)

    def current_dims(self) -> tuple[int, int]:
        return self.dims_override or (self.in_sizes[self.level], self.out_sizes[self.level])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size, out_size = self.current_dims()
        weight_part = self.linear.weight[:out_size, :in_size]
        bias_part = None
        if self.linear.bias is not None:
            bias_part = self.linear.bias[:out_size]
        return F.linear(x, weight_part, bias_part)

    def set_level_use(self, level: int) -> None:
        assert (level >= 0)
        assert (level <= self.max_level())
        self.level = level
        self.dims_override = None

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.in_sizes) - 1

    @staticmethod
    def base_type() -> type[nn.Linear]:
        return nn.Linear

    def copy_to_base(self, dest: nn.Linear) -> None:
        dest.weight.data = self.linear.weight.data[:self.out_sizes[self.level],
                                                   :self.in_sizes[self.level]]
        if self.linear.bias is not None:
            dest.bias.data = self.linear.bias.data[:self.out_sizes[self.level]]

    def load_from_base(self, src: nn.Linear) -> None:
        self.linear.weight.data[:self.out_sizes[self.level],
                                :self.in_sizes[self.level]] = src.weight.data
        if src.bias is not None:
            self.linear.bias.data[:self.out_sizes[self.level]] = src.bias.data

    def _make_reg_layer(self):
        return nn.Linear(
            self.in_sizes[self.level], self.out_sizes[self.level], *self._args, **self._kwargs)

    def export_level_delta(self) -> tuple[DownDelta[tuple[int, int]], UpDelta[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        weights = self.linear.weight.data
        lower_part = weights[:self.out_sizes[self.level],
                             self.in_sizes[self.level-1]:self.in_sizes[self.level], ]
        right_part = weights[self.out_sizes[self.level-1]                             :self.out_sizes[self.level], :self.in_sizes[self.level-1]]
        bias_part = None
        if self.linear.bias is not None:
            bias_part = self.linear.bias.data[
                self.out_sizes[self.level-1]:self.out_sizes[self.level]]
        prune_up = (lower_part, right_part, bias_part)
        prune_down = (self.in_sizes[self.level], self.out_sizes[self.level])
        return DownDelta(prune_down), UpDelta(prune_up)

    @staticmethod
    def apply_level_delta_down(model: nn.Linear, level_delta: DownDelta[tuple[int, int]]) -> None:
        in_size, out_size = level_delta.delta
        model.weight.data = model.weight.data[:out_size, :in_size].to(
            model.weight.data)
        if model.bias is not None:
            model.bias.data = model.bias.data[:out_size].to(model.bias.data)

    @staticmethod
    def apply_level_delta_up(model: nn.Linear, level_delta: UpDelta[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        weights = model.weight.data
        lower_part, right_part, bias_part = level_delta.delta
        out_size, in_size, *_ = weights.size()
        weights = F.pad(
            weights, (0, lower_part.size(1), 0, right_part.size(0)))
        weights[:, in_size:] = lower_part.to(weights)
        weights[out_size:, :in_size] = right_part.to(weights)
        if model.bias is not None:
            model.bias.data = torch.cat(
                [model.bias.data, bias_part.to(model.bias.data)])
        model.weight.data = weights
        model.zero_grad()
