"""Measurement helpers specific to the MedNest runs (the generic ones are in jetson/nestbench).

MemPoller        peak system-memory use during a block. On Jetson the GPU allocates from the same
                 RAM as the CPU, so MemTotal - MemAvailable is the honest device-level number
                 (torch's allocator peak misses the CUDA context and host-side buffers).
load_width_from_full   build one width directly from the single full-width safetensors file,
                 reading only that width's prefix slices (memory-elastic deployment: one artifact,
                 every device loads the width that fits; a switch = loading another slice).
"""
from __future__ import annotations

import glob
import json
import os
import threading
import time
from typing import Dict, Optional

import torch
from torch import nn


def _mem_used_mb() -> Optional[float]:
    try:
        mi = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if v.split():
                    mi[k] = int(v.split()[0])
        return (mi["MemTotal"] - mi["MemAvailable"]) / 1024.0
    except (OSError, KeyError, ValueError):
        return None


class MemPoller(threading.Thread):
    """with MemPoller() as mp: ...;  mp.peak_mb, mp.start_mb (None where /proc is unavailable)."""

    def __init__(self, period_s: float = 0.02):
        super().__init__(daemon=True)
        self.period_s = period_s
        self.start_mb = _mem_used_mb()
        self.peak_mb = self.start_mb
        self._halt = threading.Event()  # not _stop: that name is Thread internals

    def run(self):
        while not self._halt.is_set():
            m = _mem_used_mb()
            if m is not None and (self.peak_mb is None or m > self.peak_mb):
                self.peak_mb = m
            time.sleep(self.period_s)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self.join(timeout=2)


def drop_page_cache() -> bool:
    """Drop the OS page cache (cold-load timings). Needs passwordless sudo for `tee`; returns False
    when not allowed (the run then reports warm-cache numbers only and says so)."""
    os.sync()
    r = os.system("echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1")
    return r == 0


def _safetensors_files(model_dir: str):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        weight_map = json.load(open(idx))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in weight_map.items()}
    files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one .safetensors file or an index in {model_dir}")
    from safetensors import safe_open
    with safe_open(files[0], framework="pt") as f:
        return {k: files[0] for k in f.keys()}


@torch.no_grad()
def load_width_from_full(full_dir: str, width_config, device="cuda", dtype=torch.bfloat16) -> nn.Module:
    """A standalone model of `width_config` whose weights are the prefix slices of the full-width
    checkpoint in `full_dir`, read with safetensors slicing (only the needed bytes are touched
    for row prefixes; column prefixes read the leading rows). Equals the exported dense level."""
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    from .elastic import _module_and_name, _rebuild_meta_buffers
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(width_config, torch_dtype=dtype)
    where = _safetensors_files(full_dir)
    tied = getattr(width_config, "tie_word_embeddings", False)
    handles: Dict[str, object] = {}
    try:
        for name, p in list(m.named_parameters(remove_duplicate=False)):
            key = name
            if key not in where and tied and key == "lm_head.weight":
                key = "model.embed_tokens.weight"
            f = handles.get(where[key]) or handles.setdefault(where[key], safe_open(where[key], framework="pt"))
            sl = f.get_slice(key)
            t = sl[tuple(slice(0, s) for s in p.shape)]
            mod, attr = _module_and_name(m, name)
            # copy=True: the slice must own compact storage (a view would keep the whole tensor alive).
            t = t.to(device=device, dtype=dtype, copy=True).contiguous()
            setattr(mod, attr, nn.Parameter(t, requires_grad=False))
    finally:
        handles.clear()
    if tied:
        m.lm_head.weight = m.model.embed_tokens.weight
    _rebuild_meta_buffers(m, device)
    return m.eval()


def param_bytes(model: nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        key = p.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += p.untyped_storage().nbytes()
    return total
