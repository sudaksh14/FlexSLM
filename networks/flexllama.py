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
    qk_norm: bool = False   # True for Qwen3-family checkpoints: per-head RMSNorm on q and k
    pretrained_hf_model: Optional[str] = None
    # Fixed per-head size shared by every level. None keeps the original convention
    # head_dim = hidden_dims[l] // num_heads[l] (so num_heads * head_dim == hidden).
    # Set it for checkpoints where that does not hold, e.g. Qwen3-0.6B (hidden 1024,
    # 16 heads, head_dim 128): q/o projections then map hidden <-> num_heads * head_dim.
    head_dim: Optional[int] = None
    # Continuum options (2026-11-12; both off = the original model):
    # level_norms: per-width offsets on every residual RMSNorm (flex_modules.rmsnorm).
    # modulation: LLaMaFlex policy-aware modulation of the attention and MLP outputs,
    #   None | "scale" (foldable into o_proj/down_proj for export) | "scale_shift" (as in
    #   LLaMaFlex; the shift needs a bias that Qwen3's MLP does not have).
    level_norms: bool = False
    modulation: Optional[str] = None

    def make_model(self) -> 'FlexLLaMA':
        return FlexLLaMA(self)

    def no_prebuilt(self) -> 'FlexLLaMAConfig':
        return dataclasses.replace(self, pretrained_hf_model=None)

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1


torch.serialization.add_safe_globals([FlexLLaMAConfig])


class WidthModulation(nn.Module):
    """LLaMaFlex policy-aware modulation (Cai et al., ICLR 2025, Eq. 10):
    y_hat = y * (1 + s(e)) + t(e), with e a sinusoidal embedding of the sub-network's
    width fractions (hidden width / max and the block's own width / max) and s, t a
    two-layer MLP. The last layer is zero-initialised (identity at the start), and the
    full width is never modulated, so the max level stays the pretrained model."""

    N_FREQ = 8  # sin + cos per frequency -> 16 features per fraction (LLaMaFlex: 16)

    def __init__(self, max_dim: int, shift: bool, hidden: int = 128):
        super().__init__()
        self.max_dim, self.shift = max_dim, shift
        self.net = nn.Sequential(nn.Linear(4 * self.N_FREQ, hidden), nn.SiLU(),
                                 nn.Linear(hidden, (2 if shift else 1) * max_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("freqs", torch.pi * 2.0 ** torch.arange(self.N_FREQ), persistent=False)

    def vectors(self, hidden_frac: float, block_frac: float, d: int):
        f = torch.tensor([hidden_frac, block_frac], device=self.freqs.device, dtype=self.freqs.dtype)
        ang = (f[:, None] * self.freqs[None]).flatten()
        out = self.net(torch.cat([ang.sin(), ang.cos()]).to(self.net[0].weight.dtype))
        scale = 1 + out[:d]
        shift = out[self.max_dim:self.max_dim + d] if self.shift else None
        return scale, shift

    def forward(self, y: torch.Tensor, hidden_frac: float, block_frac: float) -> torch.Tensor:
        if hidden_frac >= 1 and block_frac >= 1:
            return y
        scale, shift = self.vectors(hidden_frac, block_frac, y.shape[-1])
        y = y * scale.to(y.dtype)
        return y + shift.to(y.dtype) if shift is not None else y


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
        qk_norm: bool = False,
        head_dim: Optional[int] = None,
        level_norms: bool = False,
        modulation: Optional[str] = None,
    ):
        super().__init__()
        self.input_layernorm          = fm.RMSNorm(hidden_dims, eps=rms_norm_eps, level_norms=level_norms)
        self.self_attn                = fm.RoPEAttention(hidden_dims, num_heads,
                                                         num_kv_heads=num_kv_heads,
                                                         rope_theta=rope_theta,
                                                         max_seq_len=max_seq_len,
                                                         qkv_bias=qkv_bias,
                                                         qk_norm=qk_norm,
                                                         rms_norm_eps=rms_norm_eps,
                                                         head_dim=head_dim)
        self.post_attention_layernorm = fm.RMSNorm(hidden_dims, eps=rms_norm_eps, level_norms=level_norms)
        self.mlp                      = LLaMAMLPBlock(hidden_dims, intermediate_dims)
        assert modulation in (None, "scale", "scale_shift"), modulation
        self.max_hidden, self.max_heads = list(hidden_dims)[-1], list(num_heads)[-1]
        self.max_intermediate = list(intermediate_dims)[-1]
        if modulation:
            self.attn_mod = WidthModulation(self.max_hidden, shift=modulation == "scale_shift")
            self.mlp_mod = WidthModulation(self.max_hidden, shift=modulation == "scale_shift")
        else:
            self.attn_mod = self.mlp_mod = None

    def width_fracs(self) -> tuple:
        """(hidden, attention, MLP) width of the current sub-network as fractions of the max."""
        return (self.input_layernorm.current_dim() / self.max_hidden,
                self.self_attn.current_num_heads() / self.max_heads,
                self.mlp.down_proj.current_dims()[0] / self.max_intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.self_attn(self.input_layernorm(x))
        if self.attn_mod is not None:
            fh, fa, _ = self.width_fracs()
            a = self.attn_mod(a, fh, fa)
        x = x + a
        m = self.mlp(self.post_attention_layernorm(x))
        if self.mlp_mod is not None:
            fh, _, fmlp = self.width_fracs()
            m = self.mlp_mod(m, fh, fmlp)
        return x + m


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
                qk_norm=getattr(config, "qk_norm", False),
                head_dim=getattr(config, "head_dim", None),
                level_norms=getattr(config, "level_norms", False),
                modulation=getattr(config, "modulation", None),
            ) for i in range(config.num_layers)}
        ))

        self.norm = fm.RMSNorm(hidden_dims, eps=config.rms_norm_eps,
                               level_norms=getattr(config, "level_norms", False))

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

    def set_widths(self, hidden: int, intermediate, heads) -> None:
        """Run at explicit widths instead of a trained level: residual width `hidden`
        (shared by every layer), and per layer an MLP width and a query-head count
        (int = the same in every layer, or one value per layer). Any prefix of the max
        widths works, so this gives untrained intermediate widths (continuum) and
        per-layer mixes (MatFormer Mix'n'Match). set_level_use undoes it."""
        if not isinstance(self.lm_head, _TiedLMHead):
            raise NotImplementedError("set_widths needs tied embeddings (Qwen3-0.6B / 1.7B are tied)")
        n = len(self.layers)
        inter = [intermediate] * n if isinstance(intermediate, int) else list(intermediate)
        heads = [heads] * n if isinstance(heads, int) else list(heads)
        assert len(inter) == n and len(heads) == n
        assert 0 < hidden <= self.hidden_dims[-1]
        self.set_level_use(self.max_level())
        self.embed_tokens.dim_override = hidden
        self.norm.dim_override = hidden
        for block, m, h in zip(self.layers, inter, heads):
            attn = block.self_attn
            nq, nkv = attn.override_head_counts(h)
            hd = attn.head_dims[-1]
            assert 0 < m <= block.max_intermediate
            block.input_layernorm.dim_override = hidden
            block.post_attention_layernorm.dim_override = hidden
            attn.heads_override = nq
            attn.q_proj.dims_override = (hidden, nq * hd)
            attn.k_proj.dims_override = (hidden, nkv * hd)
            attn.v_proj.dims_override = (hidden, nkv * hd)
            attn.o_proj.dims_override = (nq * hd, hidden)
            block.mlp.gate_proj.dims_override = (hidden, m)
            block.mlp.up_proj.dims_override = (hidden, m)
            block.mlp.down_proj.dims_override = (m, hidden)

    def set_level_use(self, level) -> None:
        super().set_level_use(level)
        for block in self.layers:
            block.self_attn.heads_override = None

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
        w = self._embedding.weight[:, :self._embedding.current_dim()]
        return torch.nn.functional.linear(x, w)
