"""Timing primitives for on-device latency measurement.

Rules this module enforces, because getting any of them wrong silently
invalidates a latency table on Jetson:

1. **Synchronise.** CUDA is asynchronous; ``time.perf_counter()`` around a
   launch measures the launch, not the work.  Every timer here brackets with
   ``torch.cuda.synchronize()`` or with CUDA events.
2. **Warm up.** The first iterations pay cuDNN/cuBLAS autotuning, lazy module
   init, and -- on Jetson specifically -- DVFS ramp-up from the idle clock.  On
   an Orin Nano the first call can be 3-5x the steady-state cost.
3. **Repeat and report a distribution.** A single number from a passively
   cooled board is a coin flip.  We keep every per-iteration sample.
4. **Watch the clock during the run.** If the GPU clock at the end of a block
   is below the clock at the start, the board throttled and the block should be
   discarded or flagged, not averaged.

The CUDA-event path is preferred for sub-millisecond regions (the vision tower,
the resampler); the wall-clock path with an explicit synchronise is used for
whole-query timings, where the launch overhead is part of what we want to
measure anyway.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def cuda_timer(store: List[float]):
    """Append the elapsed milliseconds of the body to ``store`` (CUDA events)."""
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        yield
        store.append((time.perf_counter() - t0) * 1000.0)
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    store.append(start.elapsed_time(end))


@contextmanager
def wall_timer(store: List[float]):
    """Append elapsed milliseconds, synchronising before and after."""
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    store.append((time.perf_counter() - t0) * 1000.0)


def timed_loop(fn: Callable[[], object], warmup: int = 5, iters: int = 30,
               use_events: bool = False,
               clock_probe: Optional[Callable[[], Optional[int]]] = None,
               inter_iter_sleep_s: float = 0.0) -> Dict[str, object]:
    """Run ``fn`` ``warmup`` + ``iters`` times; return samples and clock drift.

    Returns ``{"samples_ms": [...], "warmup_ms": [...], "clock_start",
    "clock_end", "throttled"}``.  ``throttled`` is True when the GPU clock fell
    by more than 5% between the first and last iteration, which on Orin Nano is
    the signature of the board hitting its thermal or power cap mid-block.
    """
    warm: List[float] = []
    timer = cuda_timer if use_events else wall_timer
    for _ in range(max(0, warmup)):
        with timer(warm):
            fn()

    clock_start = clock_probe() if clock_probe else None
    samples: List[float] = []
    for _ in range(max(1, iters)):
        with timer(samples):
            fn()
        if inter_iter_sleep_s:
            time.sleep(inter_iter_sleep_s)
    clock_end = clock_probe() if clock_probe else None

    throttled = None
    if clock_start and clock_end:
        throttled = clock_end < 0.95 * clock_start

    return {"samples_ms": samples, "warmup_ms": warm,
            "clock_start_hz": clock_start, "clock_end_hz": clock_end,
            "throttled": throttled}


# ---------------------------------------------------------------------------
# per-module attribution
# ---------------------------------------------------------------------------

class ModuleTimer:
    """Forward-hook timer that attributes wall time to named submodules.

    Used for the vision-tower / resampler+projector / language-model split the
    paper quotes (Section II: "21% of parameters, 36% of FLOPs, 30% of latency
    for NEST-VLM-TinyLlama at 256 tokens").  Attribution needs a synchronise per
    boundary, so the *totals* under a ModuleTimer are inflated relative to an
    unhooked run -- report the split as a percentage, and take the absolute
    latency from an unhooked timing pass.  Both numbers are recorded so this
    caveat is checkable rather than assumed.
    """

    def __init__(self, model, targets: Dict[str, object]):
        self.targets = {k: v for k, v in targets.items() if v is not None}
        self.records: Dict[str, List[float]] = {k: [] for k in self.targets}
        self._handles = []
        self._starts: Dict[str, float] = {}

    def _pre(self, name):
        def hook(_mod, _inp):
            sync()
            self._starts[name] = time.perf_counter()
        return hook

    def _post(self, name):
        def hook(_mod, _inp, _out):
            sync()
            t0 = self._starts.pop(name, None)
            if t0 is not None:
                self.records[name].append((time.perf_counter() - t0) * 1000.0)
        return hook

    def __enter__(self) -> "ModuleTimer":
        for name, mod in self.targets.items():
            self._handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self._handles.append(mod.register_forward_hook(self._post(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        for k in self.records:
            self.records[k] = []

    def totals_ms(self) -> Dict[str, float]:
        return {k: sum(v) for k, v in self.records.items()}

    def per_call_ms(self) -> Dict[str, float]:
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self.records.items()}


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def memory_snapshot() -> Dict[str, float]:
    """CUDA allocator view plus the process RSS.

    On Jetson, GPU and CPU share one physical pool, so the allocator's
    ``max_memory_allocated`` is a *lower bound* on what the board actually
    needed: it excludes the CUDA context (~300-600 MB on Orin), the pinned host
    buffers, and the image-preprocessing arrays on the CPU side.  The honest
    device-level number is the peak of tegrastats' RAM field over the run,
    which the experiment scripts capture alongside this.
    """
    out: Dict[str, float] = {}
    if torch.cuda.is_available():
        out["cuda_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        out["cuda_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        out["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    # read_sysfs rather than open(): one hardened reader for every /proc and
    # /sys node in the harness (see nestbench/device.py:_read).
    from .device import read_sysfs

    try:
        for line in (read_sysfs("/proc/self/status") or "").splitlines():
            if line.startswith("VmHWM:"):
                out["proc_peak_rss_gb"] = int(line.split()[1]) / 1e6
            elif line.startswith("VmRSS:"):
                out["proc_rss_gb"] = int(line.split()[1]) / 1e6
    except (OSError, ValueError, IndexError):
        pass
    try:
        mi = {}
        for line in (read_sysfs("/proc/meminfo") or "").splitlines():
            k, _, v = line.partition(":")
            if v.split():
                mi[k] = int(v.split()[0])
        if mi:
            out["sys_mem_used_gb"] = (mi.get("MemTotal", 0)
                                      - mi.get("MemAvailable", 0)) / 1e6
            out["sys_mem_total_gb"] = mi.get("MemTotal", 0) / 1e6
            out["sys_swap_used_gb"] = (mi.get("SwapTotal", 0)
                                       - mi.get("SwapFree", 0)) / 1e6
    except (OSError, ValueError, IndexError):
        pass
    return out


def gpu_clock_hz() -> Optional[int]:
    from .device import clocks_snapshot
    return clocks_snapshot().get("gpu_cur_hz")
