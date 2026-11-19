"""Elastic inference for an exported MedNest model: one set of weights, any trained width.

A MedNest width is a prefix slice of the full-width weights in every dimension (the exporter,
training/export_flex_hf.py, writes level l as exactly that, with no rescaling). So each width can
be served as a standard Hugging Face Qwen3 model whose parameters are *views* into the full-width
tensors:

    full = ElasticModel.load("hf_export/<config>/level8", level_configs=[.../level0/config.json, ...])
    m = full.width(3)        # a Qwen3ForCausalLM at width 3; shares storage with `full`
    m(input_ids)             # standard forward / KV cache; no weights copied, no extra memory

Switching width = picking another view model: nothing is loaded or copied. Two modes:
  "view"         zero-copy views (non-contiguous for column slices; cuBLAS takes the leading
                 dimension, so no copy is needed, but whether the Orin's kernels are as fast on
                 strided weights is an empirical question: bench_switch.py measures it)
  "materialize"  on switch, copy the active width into contiguous tensors (extra memory = the
                 active width; switch time = the copy). The fallback if views turn out slow.
check_views() compares a view model with the exported dense checkpoint of the same level.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

WIDTH_KEYS = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")


@dataclass
class WidthSpec:
    level: int
    config: object                    # transformers PretrainedConfig of this width
    params: int = 0                   # total parameters (tied embedding counted once)

    @property
    def dims(self) -> Dict[str, int]:
        return {k: getattr(self.config, k) for k in WIDTH_KEYS}


def _prefix(t: torch.Tensor, shape) -> torch.Tensor:
    """Prefix slice of `t` to `shape` in every dimension (a view, no copy)."""
    if tuple(t.shape) == tuple(shape):
        return t
    if len(shape) != t.dim() or any(s > d for s, d in zip(shape, t.shape)):
        raise ValueError(f"cannot slice {tuple(t.shape)} to {tuple(shape)}")
    return t[tuple(slice(0, s) for s in shape)]


def _module_and_name(model: nn.Module, qualified: str):
    path, _, name = qualified.rpartition(".")
    return (model.get_submodule(path) if path else model), name


def _rebuild_meta_buffers(model: nn.Module, device) -> None:
    """Modules built on the meta device keep meta buffers (Qwen3's RoPE inv_freq). Re-create the
    rotary embedding on `device` from the config; refuse any other meta buffer."""
    for name, mod in list(model.named_modules()):
        if type(mod).__name__.endswith("RotaryEmbedding") and any(b.is_meta for b in mod.buffers(recurse=False)):
            parent, attr = _module_and_name(model, name)
            setattr(parent, attr, type(mod)(config=model.config, device=device))
    left = [n for n, b in model.named_buffers() if b.is_meta]
    if left:
        raise RuntimeError(f"meta buffers left after building the view model: {left[:5]}")


class ElasticModel:
    """The full-width model plus one view model per width."""

    def __init__(self, full: nn.Module, specs: List[WidthSpec], mode: str = "view"):
        if mode not in ("view", "materialize"):
            raise ValueError(mode)
        self.full = full.eval()
        self.specs = sorted(specs, key=lambda s: s.level)
        self.mode = mode
        self.device = next(full.parameters()).device
        self.dtype = next(full.parameters()).dtype
        self._views: Dict[int, nn.Module] = {}
        self._active: Optional[int] = None
        self._materialized: Optional[nn.Module] = None
        full_dims = {k: getattr(full.config, k) for k in WIDTH_KEYS}
        if self.specs[-1].dims != full_dims:
            raise ValueError(f"the widest level {self.specs[-1].dims} is not the loaded model {full_dims}")

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def load(cls, full_dir: str, level_configs: List[str], device="cuda", dtype=torch.bfloat16,
             mode: str = "view") -> "ElasticModel":
        """full_dir: exported widest level (HF dir). level_configs: config.json of every level
        (paths or the level directories), the widest included."""
        from transformers import AutoConfig, AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(full_dir, torch_dtype=dtype).to(device)
        specs = []
        for i, p in enumerate(level_configs):
            d = p if os.path.isdir(p) else os.path.dirname(p)
            cfg = AutoConfig.from_pretrained(d)
            level = int(os.path.basename(os.path.normpath(d)).replace("level", "")) \
                if os.path.basename(os.path.normpath(d)).startswith("level") else i
            specs.append(WidthSpec(level=level, config=cfg))
        return cls(full, specs, mode=mode)

    @classmethod
    def from_widths(cls, full: nn.Module, widths: List[Dict[str, int]], mode: str = "view") -> "ElasticModel":
        """Widths given as dicts of WIDTH_KEYS (tests; or levels without exported configs)."""
        specs = []
        for i, w in enumerate(widths):
            cfg = full.config.__class__.from_dict(full.config.to_dict())
            for k, v in w.items():
                setattr(cfg, k, v)
            specs.append(WidthSpec(level=i, config=cfg))
        return cls(full, specs, mode=mode)

    def _build_view(self, spec: WidthSpec) -> nn.Module:
        from transformers import AutoModelForCausalLM
        cfg = spec.config
        cfg.torch_dtype = self.dtype
        # Same attention kernel as the loaded model (SDPA by default): from_config would otherwise
        # pick its own default, which changes speed and the last bits of the logits.
        attn = getattr(self.full.config, "_attn_implementation", None)
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg, torch_dtype=self.dtype,
                                                 **({"attn_implementation": attn} if attn else {}))
        full_params = dict(self.full.named_parameters(remove_duplicate=False))
        tied = getattr(cfg, "tie_word_embeddings", False)
        for name, p in list(m.named_parameters(remove_duplicate=False)):
            src = full_params.get(name)
            if src is None and tied and name == "lm_head.weight":
                src = full_params["model.embed_tokens.weight"]
            if src is None:
                raise KeyError(f"{name} not in the full model")
            mod, attr = _module_and_name(m, name)
            setattr(mod, attr, nn.Parameter(_prefix(src.detach(), p.shape), requires_grad=False))
        if tied:
            m.lm_head.weight = m.model.embed_tokens.weight
        _rebuild_meta_buffers(m, self.device)
        m.eval()
        m.generation_config = self.full.generation_config
        spec.params = sum(t.numel() for t in {id(t): t for t in m.parameters()}.values())
        return m

    def build_all(self) -> None:
        for s in self.specs:
            if s.level not in self._views:
                self._views[s.level] = self._build_view(s)

    # ---- use ----------------------------------------------------------------------------
    @property
    def levels(self) -> List[int]:
        return [s.level for s in self.specs]

    def spec(self, level: int) -> WidthSpec:
        return next(s for s in self.specs if s.level == level)

    def width(self, level: int) -> nn.Module:
        """The model for `level` (view mode: the cached view model; materialize mode: a contiguous
        copy of that width, replacing the previous one)."""
        if level not in self._views:
            self._views[level] = self._build_view(self.spec(level))
        view = self._views[level]
        if self.mode == "view":
            self._active = level
            return view
        if self._active == level and self._materialized is not None:
            return self._materialized
        self._materialized = None                      # free the previous copy first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = _contiguous_copy(view)
        self._materialized, self._active = m, level
        return m

    def switch(self, level: int) -> Dict[str, float]:
        """Switch to `level`; returns the wall time (ms, synchronised) the switch itself took."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.width(level)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"level": level, "switch_ms": (time.perf_counter() - t0) * 1000.0, "mode": self.mode}

    def unique_param_bytes(self) -> int:
        """Bytes of distinct weight storage held (views share the full model's storage)."""
        seen, total = set(), 0
        models = [self.full] + list(self._views.values()) + ([self._materialized] if self._materialized else [])
        for m in models:
            for p in m.parameters():
                key = (p.untyped_storage().data_ptr(), p.untyped_storage().nbytes())
                if key not in seen:
                    seen.add(key)
                    total += p.untyped_storage().nbytes()
        return total


def _contiguous_copy(view_model: nn.Module) -> nn.Module:
    """A copy of a view model with contiguous parameters (same config, same device)."""
    from transformers import AutoModelForCausalLM
    cfg = view_model.config
    dtype = next(view_model.parameters()).dtype
    device = next(view_model.parameters()).device
    attn = getattr(cfg, "_attn_implementation", None)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, **({"attn_implementation": attn} if attn else {}))
    src = dict(view_model.named_parameters(remove_duplicate=False))
    for name, p in list(m.named_parameters(remove_duplicate=False)):
        mod, attr = _module_and_name(m, name)
        setattr(mod, attr, nn.Parameter(src[name].detach().contiguous().clone(), requires_grad=False))
    if getattr(cfg, "tie_word_embeddings", False):
        m.lm_head.weight = m.model.embed_tokens.weight
    _rebuild_meta_buffers(m, device)
    m.generation_config = view_model.generation_config
    return m.eval()


@torch.no_grad()
def check_views(elastic: ElasticModel, level_dirs: Dict[int, str], prompts: List[str], tokenizer,
                atol: float = None) -> List[dict]:
    """Compare each view model with the exported dense checkpoint of the same level on `prompts`
    (last-position logits): max |diff| and argmax agreement. Proves the on-device elastic model is
    the evaluated model."""
    from transformers import AutoModelForCausalLM
    out = []
    for level, d in sorted(level_dirs.items()):
        if not any(f.endswith((".safetensors", ".bin")) for f in os.listdir(d)):
            continue      # config-only level (the staged export keeps weights for the anchors only)
        dense = AutoModelForCausalLM.from_pretrained(d, torch_dtype=elastic.dtype).to(elastic.device).eval()
        view = elastic.width(level)
        diffs, agree = [], 0
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to(elastic.device)
            a = view(ids).logits[0, -1].float()
            b = dense(ids).logits[0, -1].float()
            diffs.append((a - b).abs().max().item())
            agree += int(a.argmax() == b.argmax())
        out.append({"level": level, "max_abs_logit_diff": max(diffs), "argmax_agree": agree / len(prompts),
                    "n_prompts": len(prompts),
                    "same_params": sum(t.numel() for t in dense.parameters()) == elastic.spec(level).params})
        del dense
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def level_dirs_of(export_dir: str) -> Dict[int, str]:
    """{level: dir} of an export directory hf_export/<config>/level<i>."""
    out = {}
    for name in os.listdir(export_dir):
        if name.startswith("level") and name[5:].isdigit() and os.path.exists(os.path.join(export_dir, name, "config.json")):
            out[int(name[5:])] = os.path.join(export_dir, name)
    return dict(sorted(out.items()))


def describe(elastic: ElasticModel) -> List[dict]:
    rows = []
    for s in elastic.specs:
        if not s.params:
            elastic.width(s.level) if elastic.mode == "view" else elastic._build_view(s)
        rows.append({"level": s.level, **s.dims, "params": s.params})
    return rows


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
