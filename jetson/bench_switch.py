#!/usr/bin/env python3
"""What changing the width costs on the device, against the alternatives.

  A  elastic, view mode      switch = pick another view model (weights shared, nothing copied)
  B  elastic, materialize    switch = copy the active width into contiguous tensors
  C  separate models         switch = load another exported model from disk (from_pretrained),
                             page cache warm and (with passwordless sudo) cold
  D  one file, slice load    switch = read only the new width's prefix slices from the single
                             full-width checkpoint (memory-elastic deployment)
  E  memory                  weights resident for elastic (one model) vs all separate models
                             resident at once (the only zero-switch-cost alternative)

For A and B every switch is followed by one fixed short request; its time to first token is
recorded (the first request at a new width also pays any kernel selection for new shapes) and
compared with the same request repeated at that width.

    python3 jetson/bench_switch.py --elastic models/mednest --levels 0 4 8 --switches 60
"""
import gc
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mednest_jetson.cli import base_parser, cooldown, setup, torch_dtype  # noqa: E402


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ttft(model, ids):
    import torch
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(input_ids=ids).logits[:, -1].argmax(-1).item()
    return (time.perf_counter() - t0) * 1000.0


def switch_sequence(levels, n, seed=0):
    rng, seq, cur = random.Random(seed), [], None
    for _ in range(n):
        nxt = rng.choice([l for l in levels if l != cur] or levels)
        seq.append(nxt)
        cur = nxt
    return seq


def elastic_switches(w, args, mode, dirs, levels, ids, dtype):
    from mednest_jetson import elastic as el
    from mednest_jetson.measure import MemPoller
    from nestbench.results import stat_block
    cooldown(args)
    with MemPoller() as mp:
        t0 = time.perf_counter()
        e = el.ElasticModel.load(dirs[max(dirs)], [dirs[l] for l in sorted(dirs)], device=args.device, dtype=dtype, mode=mode)
        load_ms = (time.perf_counter() - t0) * 1000.0
        build = {}
        for l in levels:                     # first use of each width: builds its view (once)
            _sync()
            t1 = time.perf_counter()
            e.width(l)
            _sync()
            build[l] = (time.perf_counter() - t1) * 1000.0
        for l in levels:                     # warm every width once
            ttft(e.width(l), ids)
        recs = []
        for l in switch_sequence(levels, args.switches):
            s = e.switch(l)
            first = ttft(e.width(l), ids)
            again = ttft(e.width(l), ids)
            rec = {"kind": "switch", "path": f"elastic_{mode}", "to_level": l, "switch_ms": s["switch_ms"],
                   "ttft_after_switch_ms": first, "ttft_repeat_ms": again}
            w.write(rec)
            recs.append(rec)
    summ = {"kind": "switch_summary", "path": f"elastic_{mode}", "load_full_ms": load_ms, "first_build_ms": build,
            "switch_ms": stat_block([r["switch_ms"] for r in recs]),
            "ttft_after_switch_ms": stat_block([r["ttft_after_switch_ms"] for r in recs]),
            "ttft_repeat_ms": stat_block([r["ttft_repeat_ms"] for r in recs]),
            "weight_bytes_resident": e.unique_param_bytes(), "peak_sys_mem_mb": mp.peak_mb, "sys_mem_before_mb": mp.start_mb}
    w.write(summ)
    print(f"  {summ['path']}: switch median {summ['switch_ms']['median']:.3f} ms, TTFT after switch "
          f"{summ['ttft_after_switch_ms']['median']:.1f} vs repeat {summ['ttft_repeat_ms']['median']:.1f} ms, "
          f"resident weights {summ['weight_bytes_resident'] / 1e9:.2f} GB", flush=True)
    del e
    gc.collect()


def load_timings(w, args, path, loader, levels, ids, can_drop):
    import torch

    from mednest_jetson.measure import MemPoller, drop_page_cache, param_bytes
    for l in levels:
        for cache in (["cold"] if can_drop else []) + ["warm"] * args.load_repeats:
            cooldown(args)
            if cache == "cold":
                drop_page_cache()
            gc.collect()
            torch.cuda.empty_cache()
            with MemPoller() as mp:
                _sync()
                t0 = time.perf_counter()
                m = loader(l)
                _sync()
                load_ms = (time.perf_counter() - t0) * 1000.0
                first = ttft(m, ids)
            rec = {"kind": "load", "path": path, "level": l, "cache": cache, "load_ms": load_ms,
                   "ttft_first_ms": first, "load_plus_first_ms": load_ms + first,
                   "weight_bytes": param_bytes(m), "peak_sys_mem_mb": mp.peak_mb, "sys_mem_before_mb": mp.start_mb}
            w.write(rec)
            print(f"  {path} level{l} ({cache}): load {load_ms:.0f} ms + first token {first:.0f} ms, "
                  f"{rec['weight_bytes'] / 1e9:.2f} GB", flush=True)
            del m


def main():
    p = base_parser(__doc__)
    p.add_argument("--elastic", required=True, help="MedNest export dir (level<i>/ subdirs)")
    p.add_argument("--levels", type=int, nargs="*", default=None, help="widths to switch among (default: all)")
    p.add_argument("--switches", type=int, default=60)
    p.add_argument("--load-repeats", type=int, default=2, help="warm-cache loads per level")
    p.add_argument("--prompt-tokens", type=int, default=256, help="length of the fixed request")
    p.add_argument("--skip", nargs="*", default=[], choices=["A", "B", "C", "D", "E"])
    args = p.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from mednest_jetson import elastic as el
    from mednest_jetson.measure import drop_page_cache, load_width_from_full

    w = setup(args, "switch")
    dtype = torch_dtype(args.dtype)
    dirs = el.level_dirs_of(args.elastic)
    levels = args.levels if args.levels is not None else sorted(dirs)
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (1, args.prompt_tokens), device=args.device)
    can_drop = drop_page_cache()
    w.write({"kind": "info", "levels": levels, "cold_cache_possible": can_drop})
    if not can_drop:
        print("  [switch] no passwordless sudo for drop_caches: load timings are warm-cache only", flush=True)

    if "A" not in args.skip:
        elastic_switches(w, args, "view", dirs, levels, ids, dtype)
    if "B" not in args.skip:
        elastic_switches(w, args, "materialize", dirs, levels, ids, dtype)
    if "C" not in args.skip:
        load_timings(w, args, "separate_model", lambda l: AutoModelForCausalLM.from_pretrained(
            dirs[l], torch_dtype=dtype).to(args.device).eval(), levels, ids, can_drop)
    if "D" not in args.skip:
        cfgs = {l: AutoConfig.from_pretrained(dirs[l]) for l in levels}
        load_timings(w, args, "slice_from_full", lambda l: load_width_from_full(
            dirs[max(dirs)], cfgs[l], device=args.device, dtype=dtype), levels, ids, can_drop)
    if "E" not in args.skip:
        from mednest_jetson.measure import MemPoller, param_bytes
        gc.collect()
        torch.cuda.empty_cache()
        with MemPoller() as mp:
            ms = [AutoModelForCausalLM.from_pretrained(dirs[l], torch_dtype=dtype).to(args.device) for l in levels]
        allres = sum(param_bytes(m) for m in ms)
        del ms
        gc.collect()
        torch.cuda.empty_cache()
        bits = torch.finfo(dtype).bits
        e = el.ElasticModel.load(dirs[max(dirs)], [dirs[l] for l in sorted(dirs)], device=args.device, dtype=dtype)
        e.build_all()
        rec = {"kind": "memory", "levels": levels, "all_separate_resident_bytes": allres,
               "all_separate_peak_sys_mem_mb": mp.peak_mb, "sys_mem_before_mb": mp.start_mb,
               "elastic_resident_bytes": e.unique_param_bytes(),
               "all_9_separate_bytes_computed": sum(s.params for s in e.specs) * bits // 8}
        w.write(rec)
        print(f"  memory: elastic {rec['elastic_resident_bytes'] / 1e9:.2f} GB for {len(e.specs)} widths vs "
              f"{allres / 1e9:.2f} GB for {len(levels)} separate models resident "
              f"({rec['all_9_separate_bytes_computed'] / 1e9:.2f} GB for all {len(e.specs)})", flush=True)
    w.close()


if __name__ == "__main__":
    main()
