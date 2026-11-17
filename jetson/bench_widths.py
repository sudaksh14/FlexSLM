#!/usr/bin/env python3
"""Per-width cost and accuracy on the device: for every elastic width (a view of the one set of
weights) and every dense baseline, answer the same stratified test subset one question at a time
(batch 1, greedy, KV cache) and record

  latency    time to first token, decode ms/token, end-to-end ms per answer (median, p95)
  memory     weight bytes (resident full weights; the active width's own slice), peak system RAM
             in use (unified memory) and the CUDA allocator peak
  energy     module input power (VDD_IN) integrated over the block; gross and net of the idle floor
  accuracy   answers scored exactly as on the cluster (eval/eval_gen.py), per benchmark

    python3 jetson/bench_widths.py --elastic models/mednest --levels 0 4 8 \
        --dense conventional=models/qwen3-0.6b-conventional \
        --dense apollo2-0.5b=models/Apollo2-0.5B:apollo_answer:16 \
        --dense qwen3-0.6b-reasoning=models/Qwen3-0.6B:as_is:1024 --power-mode 15W

--dense NAME=DIR[:FORMAT[:MAX_NEW_TOKENS]]  FORMAT as eval_gen --prompt_format (default as_is, 32).
Baselines with more than 64 new tokens (untrained models that reason) answer only --long-limit
questions per benchmark, since each takes ~1,000 decode steps.
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mednest_jetson.cli import base_parser, cooldown, setup, torch_dtype  # noqa: E402


def parse_dense(spec: str):
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    return {"name": name, "dir": parts[0], "fmt": parts[1] if len(parts) > 1 and parts[1] else "as_is",
            "max_new": int(parts[2]) if len(parts) > 2 else 32}


def per_source_limit(rows, k):
    out, seen = [], {}
    for r in rows:
        if seen.get(r["source"], 0) < k:
            out.append(r)
            seen[r["source"]] = seen.get(r["source"], 0) + 1
    return out


def run_block(w, args, target: dict, model, tokenizer, rows, idle_w, extra: dict, decoder=None):
    import torch

    from mednest_jetson import workload as wl
    from mednest_jetson.measure import MemPoller
    from nestbench import device
    from nestbench.power import PowerSampler
    from nestbench.results import stat_block
    from nestbench.timing import gpu_clock_hz, memory_snapshot, reset_peak_memory

    cd = cooldown(args)
    prepare_ms = 0.0
    if decoder is not None:   # views, caches and CUDA graphs for every cache length these questions need
        needs = {wl.encode(tokenizer, wl.prompt_for(r, target["fmt"], tokenizer), target["fmt"], "cpu").shape[1]
                 + target["max_new"] for r in rows}
        prepare_ms = sum(decoder.prepare(n) for n in sorted(needs))
    for r in rows[:args.warmup]:                                   # warm-up: not recorded
        wl.answer(model, tokenizer, r, target["fmt"], min(target["max_new"], 32), args.device, decoder)
    reset_peak_memory()
    clock0, recs = gpu_clock_hz(), []
    with MemPoller() as mp, PowerSampler(source=args.power_source, period_ms=args.power_period_ms) as ps:
        t0 = time.perf_counter()
        for i, r in enumerate(rows):
            rec = wl.answer(model, tokenizer, r, target["fmt"], target["max_new"], args.device, decoder)
            rec.update(kind="answer", target=target["name"], **extra)
            w.write(rec)
            recs.append(rec)
            if (i + 1) % 20 == 0:
                print(f"    {target['name']}: {i + 1}/{len(rows)}  median total "
                      f"{stat_block([x['total_ms'] for x in recs])['median']:.0f} ms", flush=True)
        wall_s = time.perf_counter() - t0
    clock1 = gpu_clock_hz()
    pw = ps.summary()
    gross = pw.get("energy_j")
    net = (gross - idle_w * pw["duration_s"]) if (gross is not None and idle_w is not None) else None
    summ = {
        "kind": "summary", "target": target["name"], "fmt": target["fmt"], "max_new_tokens": target["max_new"],
        **extra, "n": len(recs), "wall_s": wall_s, "decoder": args.decode, "prepare_ms": prepare_ms,
        "kv_cache_bytes": decoder.kv_bytes() if decoder is not None else None,
        "accuracy": wl.accuracy_by_source(recs),
        "ttft_ms": stat_block([x["ttft_ms"] for x in recs]),
        "ms_per_decode_token": stat_block([x["ms_per_decode_token"] for x in recs if x["ms_per_decode_token"]]),
        "total_ms": stat_block([x["total_ms"] for x in recs]),
        "prompt_tokens": stat_block([x["prompt_tokens"] for x in recs]),
        "new_tokens": stat_block([x["n_new"] for x in recs]),
        "energy_gross_j": gross, "energy_net_j": net,
        "energy_gross_j_per_answer": gross / len(recs) if gross is not None else None,
        "energy_net_j_per_answer": net / len(recs) if net is not None else None,
        "mean_power_w": pw.get("mean_power_w"), "idle_power_w": idle_w, "power": pw,
        "peak_sys_mem_mb": mp.peak_mb, "sys_mem_before_mb": mp.start_mb, **{f"mem_{k}": v for k, v in memory_snapshot().items()},
        "gpu_clock_start_hz": clock0, "gpu_clock_end_hz": clock1,
        "throttled": (clock1 < 0.95 * clock0) if (clock0 and clock1) else None,
        "max_temp_c": device.max_temp_c(), "cooldown": cd,
    }
    w.write(summ)
    acc = summ["accuracy"].get("Avg")
    print(f"  == {target['name']} {extra}: acc {acc:.1f}  TTFT {summ['ttft_ms'].get('median', 0):.0f} ms  "
          f"decode {summ['ms_per_decode_token'].get('median', 0):.1f} ms/tok  total {summ['total_ms']['median']:.0f} ms  "
          f"E/answer {summ['energy_net_j_per_answer'] if net is None else round(summ['energy_net_j_per_answer'], 3)} J (net)",
          flush=True)
    return summ


def main():
    p = base_parser(__doc__)
    p.add_argument("--elastic", default=None, help="MedNest export dir (level<i>/ subdirs)")
    p.add_argument("--levels", type=int, nargs="*", default=None, help="widths to run (default: all)")
    p.add_argument("--mode", default="view", choices=["view", "materialize"])
    p.add_argument("--dense", action="append", default=[], help="NAME=DIR[:FORMAT[:MAX_NEW]] (repeatable)")
    p.add_argument("--subset", default=os.path.join(os.path.dirname(__file__), "questions", "subset.jsonl"))
    p.add_argument("--limit", type=int, default=None, help="first N subset questions (smoke tests)")
    p.add_argument("--long-limit", type=int, default=5, help="questions per benchmark for >64-token baselines")
    p.add_argument("--idle-s", type=float, default=10.0)
    p.add_argument("--decode", default="graph", choices=["graph", "eager"],
                   help="graph: static KV cache + CUDA-graph decode steps (deployment-style; eager decoding "
                        "is launch-bound at batch 1 and hides the effect of width); eager: plain HF")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mednest_jetson import elastic as el
    from mednest_jetson import workload as wl
    from mednest_jetson.decode import DenseDecoders, ElasticDecoders, max_len_for
    from mednest_jetson.measure import param_bytes
    from nestbench.power import measure_idle

    w = setup(args, "widths")
    rows = wl.load_subset(args.subset, args.limit)
    print(f"[widths] {len(rows)} questions; measuring the idle floor ({args.idle_s:.0f} s)...", flush=True)
    idle = measure_idle(source=args.power_source, seconds=args.idle_s, period_ms=args.power_period_ms)
    w.write({"kind": "idle", **idle})
    idle_w = idle.get("idle_power_w")
    dtype = torch_dtype(args.dtype)

    if args.elastic:
        dirs = el.level_dirs_of(args.elastic)
        top = max(dirs)
        e = el.ElasticModel.load(dirs[top], [dirs[l] for l in sorted(dirs)], device=args.device, dtype=dtype,
                                 mode=args.mode)
        tok = AutoTokenizer.from_pretrained(dirs[top])
        resident = e.unique_param_bytes()
        decs = None
        if args.decode == "graph":
            lens = [wl.encode(tok, wl.prompt_for(r, "as_is"), "as_is", "cpu").shape[1] for r in rows]
            decs = ElasticDecoders(e, max_len_for(lens, 32))      # one full-width KV buffer for all widths
        for level in (args.levels if args.levels is not None else e.levels):
            m = e.width(level)
            dec = decs.get(level) if decs is not None else None
            if dec is not None and args.mode == "materialize":
                dec = DenseDecoders(m, decs.max_len)               # contiguous copy: its own caches
            run_block(w, args, {"name": "mednest", "fmt": "as_is", "max_new": 32}, m, tok, rows, idle_w,
                      {"level": level, "mode": args.mode, "params": e.spec(level).params, **e.spec(level).dims,
                       "weight_bytes_resident": e.unique_param_bytes(), "weight_bytes_full": resident,
                       "weight_bytes_width": param_bytes(m) if args.mode == "materialize" else
                       e.spec(level).params * torch.finfo(dtype).bits // 8}, decoder=dec)
        del e, tok, decs
        gc.collect()
        torch.cuda.empty_cache()

    for spec in map(parse_dense, args.dense):
        tok = AutoTokenizer.from_pretrained(spec["dir"], padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|endoftext|>"
        remote = os.path.exists(os.path.join(spec["dir"], "modeling_qwen.py"))  # Apollo-1.8B (Qwen-1 code, reviewed)
        m = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=dtype, trust_remote_code=remote).to(args.device).eval()
        sub = per_source_limit(rows, args.long_limit) if spec["max_new"] > 64 else rows
        dec = None
        if args.decode == "graph":
            lens = [wl.encode(tok, wl.prompt_for(r, spec["fmt"], tok), spec["fmt"], "cpu").shape[1] for r in sub]
            dec = DenseDecoders(m, max_len_for(lens, spec["max_new"]))
        run_block(w, args, spec, m, tok, sub, idle_w,
                  {"level": None, "dir": spec["dir"], "params": sum(t.numel() for t in m.parameters()),
                   "weight_bytes_resident": param_bytes(m), "weight_bytes_width": param_bytes(m)}, decoder=dec)
        del m, tok, dec
        gc.collect()
        torch.cuda.empty_cache()

    idle2 = measure_idle(source=args.power_source, seconds=args.idle_s, period_ms=args.power_period_ms)
    w.write({"kind": "idle_end", **idle2})
    w.close()


if __name__ == "__main__":
    main()
