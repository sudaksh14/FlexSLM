from typing import Iterable, Any

from torch import nn
import torch

from flex_modules.module import Module, UpDelta, DownDelta
import torch.nn.functional as F


class SelfAttention(Module):
    def __init__(self, token_size: Iterable[int], heads: Iterable[int], dropout=0.0, is_causal=False):
        super().__init__()

        def is_positive(x): return x > 0
        assert (len(token_size) > 0)
        assert (len(token_size) == len(heads))
        assert (all(map(is_positive, token_size)))
        assert (all(map(is_positive, token_size)))
        assert (all(i % j == 0 for i, j in zip(token_size, heads)))
        head_dims = [i // j for i, j in zip(token_size, heads)]
        assert (sorted(head_dims) == head_dims)
        assert (max(token_size) == token_size[-1])
        assert (max(heads) == heads[-1])
        assert (max(head_dims) == head_dims[-1])
        assert (dropout >= 0.0)

        self.token_size = token_size
        self.proj_token_size = token_size
        self.heads = heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.max_heads = max(heads)
        self.max_token_size = max(token_size)
        self.level = self.max_level()

        self.in_weights = nn.Parameter(
            torch.rand(3 * self.max_token_size, self.max_token_size))
        self.in_bias = nn.Parameter(torch.rand(3 * self.max_token_size))
        self.out_weights = nn.Parameter(
            torch.rand(self.max_token_size, self.max_token_size))
        self.out_bias = nn.Parameter(torch.rand(self.max_token_size))

    def forward(self, x: torch.Tensor):
        target_heads = self.heads[self.current_level()]
        target_token = self.token_size[self.current_level()]
        target_hs = target_token // target_heads
        hs_max = self.max_token_size // self.max_heads

        # Calculate part of parameters used in this level
        w_in = self.in_weights.view(
            3, self.max_heads, hs_max, self.max_token_size)
        w_in = w_in[:, :target_heads, :target_hs, :target_token]
        w_in = w_in.reshape(3 * target_token, target_token)

        b_in = self.in_bias.view(3, self.max_heads, hs_max)
        b_in = b_in[:, :target_heads, :target_hs]
        b_in = b_in.reshape(3 * target_token)

        w_out = self.out_weights.view(
            self.max_token_size, self.max_heads, hs_max)
        w_out = w_out[:target_token, :target_heads, :target_hs]
        w_out = w_out.reshape(target_token, target_token)

        b_out = self.out_bias[:target_token]

        # compute self attention
        x = x.transpose(1, 0)
        attn_mask = None
        if self.is_causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                x.shape[0], device=x.device, dtype=x.dtype)
        attn, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=self.token_size[self.current_level()],
            num_heads=self.heads[self.current_level()],
            in_proj_weight=w_in,
            in_proj_bias=b_in,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout,
            out_proj_weight=w_out,
            out_proj_bias=b_out,
            training=self.training,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=self.is_causal
        )
        return attn.transpose(1, 0)

    def set_level_use(self, level: int) -> None:
        assert (level >= 0)
        assert (level <= self.max_level())
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.token_size) - 1

    @staticmethod
    def base_type() -> type[nn.MultiheadAttention]:
        return nn.MultiheadAttention

    @torch.no_grad()
    def copy_to_base(self, dest: nn.MultiheadAttention) -> None:
        target_heads = self.heads[self.current_level()]
        target_token = self.token_size[self.current_level()]
        target_hs = target_token // target_heads
        hs_max = self.max_token_size // self.max_heads

        w_in = self.in_weights.view(
            3, self.max_heads, hs_max, self.max_token_size)
        w_in = w_in[:, :target_heads, :target_hs, :target_token]
        w_in = w_in.reshape(3 * target_token, target_token)
        dest.in_proj_weight.data[:] = w_in.detach()

        b_in = self.in_bias.view(3, self.max_heads, hs_max)
        b_in = b_in[:, :target_heads, :target_hs]
        b_in = b_in.reshape(3 * target_token)
        dest.in_proj_bias.data[:] = b_in.detach()

        w_out = self.out_weights.view(
            self.max_token_size, self.max_heads, hs_max)
        w_out = w_out[:target_token, :target_heads, :target_hs]
        w_out = w_out.reshape(target_token, target_token)
        dest.out_proj.weight.data[:] = w_out.detach()

        b_out = self.out_bias[:target_token]
        dest.out_proj.bias.data[:] = b_out.detach()

    @torch.no_grad()
    def load_from_base(self, src: nn.MultiheadAttention) -> None:
        target_heads = self.heads[self.current_level()]
        target_token = self.token_size[self.current_level()]
        target_hs = target_token // target_heads
        hs_max = self.max_token_size // self.max_heads

        w_in = self.in_weights.view(
            3, self.max_heads, hs_max, self.max_token_size)
        w_in = w_in[:, :target_heads, :target_hs, :target_token]
        w_in_src = src.in_proj_weight.data
        w_in_src = w_in_src.view(3, target_heads, target_hs, target_token)
        w_in[:] = w_in_src.detach()

        b_in = self.in_bias.view(3, self.max_heads, hs_max)
        b_in = b_in[:, :target_heads, :target_hs]
        b_in_src = src.in_proj_bias.data
        b_in_src = b_in_src.view(3, target_heads, target_hs)
        b_in[:] = b_in_src.detach()

        w_out = self.out_weights.view(
            self.max_token_size, self.max_heads, hs_max)
        w_out = w_out[:target_token, :target_heads, :target_hs]
        w_out_src = src.out_proj.weight.data
        w_out_src = w_out_src.view(target_token, target_heads, target_hs)
        w_out[:] = w_out_src.detach()

        b_out = self.out_bias[:target_token]
        b_out_src = src.out_proj.bias.data
        b_out[:] = b_out_src.detach()

    def _make_reg_layer(self):
        return nn.MultiheadAttention(
            self.token_size[self.level],
            self.heads[self.level],
            dropout=self.dropout,
            batch_first=True
        )

    @torch.no_grad()
    def export_level_delta(self) -> tuple[DownDelta[tuple[int, int]], UpDelta[tuple[torch.Tensor, ...]]]:
        target_level = self.current_level()
        cur_level = target_level - 1

        hs_max = self.max_token_size // self.max_heads
        hs_target = self.token_size[target_level] // self.heads[target_level]
        hs_curr = self.token_size[cur_level] // self.heads[cur_level]

        # delta up
        # in weights
        qkv = self.in_weights.data.view(
            3, self.max_heads, hs_max, self.max_token_size)
        qkv = qkv[
            :, :self.heads[target_level], :hs_target, :self.token_size[target_level]]
        target_heads_inw = qkv[:, self.heads[cur_level]:].detach()
        curr_right_inw = qkv[
            :, :self.heads[cur_level], :, self.token_size[cur_level]:].detach()
        curr_bottom_inw = qkv[
            :, :self.heads[cur_level], hs_curr:, :self.token_size[cur_level]].detach()

        # in bias
        qkv = self.in_bias.data.view(3, self.max_heads, hs_max)
        qkv = qkv[:, :self.heads[target_level], :hs_target]
        target_heads_inb = qkv[:, self.heads[cur_level]:].detach()
        slimmed_heads_inb = qkv[:, :self.heads[cur_level], hs_curr:].detach()

        # out weights
        qkv = self.out_weights.view(
            self.max_token_size, self.max_heads, hs_max)
        qkv = qkv[
            :self.token_size[target_level], :self.heads[target_level], :hs_target]
        target_heads_ow = qkv[:, self.heads[cur_level]:]
        curr_right_ow = qkv[
            self.token_size[cur_level]:, :self.heads[cur_level], :]
        curr_bottom_ow = qkv[
            :self.token_size[cur_level], :self.heads[cur_level], hs_curr:]

        # out bias
        out_bias = self.out_bias[
            self.token_size[cur_level]:self.token_size[target_level]]

        delta_up = (
            target_heads_inw, curr_right_inw, curr_bottom_inw, target_heads_inb,
            slimmed_heads_inb, target_heads_ow, curr_right_ow, curr_bottom_ow, out_bias
        )

        delta_down = (self.token_size[target_level], self.heads[target_level])

        return DownDelta(delta_down), UpDelta(delta_up)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(b: nn.MultiheadAttention, level_delta: DownDelta[tuple[int, int]]) -> None:
        ntoken, nheads = level_delta.delta
        nhs = ntoken // nheads

        binws = b.in_proj_weight.data.view(
            3, b.num_heads, b.head_dim, b.embed_dim)
        binws = binws[:, :nheads, :nhs, :ntoken]
        binws = binws.reshape(3 * ntoken, ntoken)
        b.in_proj_weight.data = binws.detach()

        binbs = b.in_proj_bias.data.view(3, b.num_heads, b.head_dim)
        binbs = binbs[:, :nheads, :nhs]
        binbs = binbs.reshape(3 * ntoken)
        b.in_proj_bias.data = binbs.detach()

        bows = b.out_proj.weight.data.view(
            b.embed_dim, b.num_heads, b.head_dim)
        bows = bows[:ntoken, :nheads, :nhs]
        bows = bows.reshape(ntoken, ntoken)
        b.out_proj.weight.data = bows.detach()

        bobs = b.out_proj.bias.data[:ntoken]
        b.out_proj.bias.data = bobs.detach()

        b.embed_dim = ntoken
        b.num_heads = nheads
        b.head_dim = nhs

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(b: nn.MultiheadAttention, level_delta: UpDelta[tuple[torch.Tensor, ...]]) -> None:
        (
            target_heads_inw, curr_right_inw, curr_bottom_inw,
            target_heads_inb, slimmed_heads_inb,
            target_heads_ow, curr_right_ow, curr_bottom_ow,
            out_bias
        ) = level_delta.delta

        nheads = b.num_heads + target_heads_inw.shape[1]
        nhead_dim = b.head_dim + curr_bottom_inw.shape[2]
        nembed_dim = b.embed_dim + curr_right_inw.shape[3]

        t = b.in_proj_weight.view(3, b.num_heads, b.head_dim, b.embed_dim)
        t = torch.cat([t, curr_bottom_inw.to(t)], dim=2)
        t = torch.cat([t, curr_right_inw.to(t)], dim=3)
        t = torch.cat([t, target_heads_inw.to(t)], dim=1)
        t = t.view(3 * nembed_dim, nembed_dim)
        b.in_proj_weight.data = t.detach()

        t = b.in_proj_bias.data.view(3, b.num_heads, b.head_dim)
        t = torch.cat([t, slimmed_heads_inb.to(t)], dim=2)
        t = torch.cat([t, target_heads_inb.to(t)], dim=1)
        t = t.view(3 * nembed_dim)
        b.in_proj_bias.data = t.detach()

        t = b.out_proj.weight.data.view(b.embed_dim, b.num_heads, b.head_dim)
        t = torch.cat([t, curr_bottom_ow.to(t)], dim=2)
        t = torch.cat([t, curr_right_ow.to(t)], dim=0)
        t = torch.cat([t, target_heads_ow.to(t)], dim=1)
        t = t.view(nembed_dim, nembed_dim)
        b.out_proj.weight.data = t.detach()

        t = torch.cat([b.out_proj.bias.data, out_bias.to(t)])
        b.out_proj.bias.data = t.detach()

        b.num_heads = nheads
        b.head_dim = nhead_dim
        b.embed_dim = nembed_dim
