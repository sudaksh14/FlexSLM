"""Do two separate models stay resident together on the Jetson, and does that cost speed?  (board only; latency/memory, NO accuracy)

    python3 controller/coresident_test.py --case pair [--load] [--n 28] --out ~/mednest-results/coresident/<case>.jsonl
One case per process (run_coresident.sh runs them all). Random seeded weights are built on the GPU from the config, no checkpoint,
as in jetson_runner --random-from. Same decoder as the benchmarks (static cache, CUDA graphs, forced 32 tokens), requests from
jetson/questions/subset.jsonl incl. the longest prompts; the request sequence alternates models: [A, A, B, B] per question, so
the second call of each pair is the same-model repeat and the first is the request after a switch.

Cases   small  596M alone            big   1.72B alone
        pair   596M + 1.72B          pair3 1.72B + nested 0.6B (188M / 364M / 596M as views)
        both   nested 1.7B (0.50B / 1.33B / 1.72B shapes of the 9-width profile) + nested 0.6B (3 views)
--load  adds the runner's co-running CUDA load (60% duty, its own CUDA context) while requests run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "jetson"))
os.environ["MEDNEST_FORCE_LEN"] = "1"

# (hidden, heads, kv, mlp) per width; last = the full width
W06 = {"188M": (512, 8, 4, 1536), "364M": (768, 12, 6, 2304), "596M": (1024, 16, 8, 3072)}
W17 = {"0.50B": (1024, 8, 4, 3072), "1.33B": (1792, 14, 7, 5376), "1.72B": (2048, 16, 8, 6144)}


def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k, _, v = l.partition(":")
        d[k] = int(v.split()[0]) / 1024.0
    return {"avail_mb": d["MemAvailable"], "free_mb": d["MemFree"], "swap_used_mb": d["SwapTotal"] - d["SwapFree"]}


def vmstat():
    d = dict(l.split() for l in open("/proc/vmstat"))
    return {k: int(d[k]) for k in ("pswpin", "pswpout", "pgmajfault")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["small", "big", "pair", "pair3", "both"])
    ap.add_argument("--models", default=os.path.expanduser("~/mednest/models"))
    ap.add_argument("--subset", default=os.path.join(REPO, "jetson", "questions", "subset.jsonl"))
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--lean", action="store_true", help="last-position logits only (the benchmarks' decoder computes logits of "
                    "every prompt position, 0.6 GB for a 2,009-token prompt) and file pages evicted after the models are built "
                    "(the CUDA allocator does not reclaim the page cache)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from mednest_jetson import elastic as el
    from mednest_jetson import workload as wl
    from mednest_jetson.decode import DenseDecoders, ElasticDecoders, max_len_for
    from mednest_jetson.measure import MemPoller

    out = open(args.out, "w")

    def rec(**k):
        out.write(json.dumps(k) + "\n")
        out.flush()
        print({a: (round(b, 1) if isinstance(b, float) else b) for a, b in k.items()}, flush=True)

    def build(name, widths):
        d = os.path.join(args.models, name)
        cfg = AutoConfig.from_pretrained(d)
        top = list(widths.values())[-1]
        for k, v in zip(("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size"), top):
            assert getattr(cfg, k) == v, (name, k, getattr(cfg, k), v)
        cfg.tie_word_embeddings = True          # both MedNest exports tie the embedding
        torch.manual_seed(0)
        with torch.device("cuda"):
            full = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).eval()
        full.generation_config = GenerationConfig.from_pretrained(d)
        ws = [dict(hidden_size=h, num_attention_heads=a, num_key_value_heads=k, intermediate_size=m) for h, a, k, m in widths.values()]
        e = el.ElasticModel.from_widths(full, ws)
        e.build_all()
        if args.lean:
            for m in e._views.values():
                f = m.forward
                m.forward = lambda *a, _f=f, **k: _f(*a, **{**k, "logits_to_keep": 1})
        return e, list(widths), AutoTokenizer.from_pretrained(d)

    full17, full06 = {"1.72B": W17["1.72B"]}, {"596M": W06["596M"]}
    spec = {"small": [("Qwen3-0.6B", full06)], "big": [("Qwen3-1.7B", full17)],
            "pair": [("Qwen3-1.7B", full17), ("Qwen3-0.6B", full06)], "pair3": [("Qwen3-1.7B", full17), ("Qwen3-0.6B", W06)],
            "both": [("Qwen3-1.7B", W17), ("Qwen3-0.6B", W06)]}[args.case]
    if args.lean:
        import subprocess
        evict = [sys.executable, os.path.join(REPO, "jetson", "drop_file_cache.py"), args.models,
                 os.path.expanduser("~/mednest-results"), os.path.expanduser("~/.cache"), os.environ.get("VIRTUAL_ENV", "/tmp")]
    rec(kind="start", case=args.case, load=args.load, lean=args.lean, **meminfo(), **vmstat())
    load = None
    try:
        rows = [json.loads(l) for l in open(args.subset, encoding="utf8")]
        mods = []
        for name, widths in spec:
            e, labels, tok = build(name, widths)
            torch.cuda.empty_cache()
            mods.append((e, labels, tok))
            rec(kind="built", model=name, widths=labels, cuda_alloc_gb=torch.cuda.memory_allocated() / 1e9,
                cuda_reserved_gb=torch.cuda.memory_reserved() / 1e9, **meminfo())
        if args.lean:
            subprocess.run(evict, check=True)
            torch.cuda.empty_cache()
            rec(kind="evicted", **meminfo())
        tok = mods[0][2]
        enc = [wl.encode(tok, wl.prompt_for(r, "as_is"), "as_is", "cpu") for r in rows]
        plen = [int(x.shape[1]) for x in enc]
        long4 = sorted(range(len(rows)), key=lambda i: -plen[i])[:4]
        step = max(1, len(rows) // (args.n - 4))
        pick = list(range(0, len(rows), step))[:args.n - 4] + long4
        rec(kind="requests", n=len(pick), longest=max(plen), median=sorted(plen)[len(plen) // 2])
        needs = sorted({plen[i] + 32 for i in pick})
        decs = []
        ml = max_len_for([plen[i] for i in pick], 32)
        for e, labels, _ in mods:
            d = ElasticDecoders(e, ml, use_graph=True)
            ms = d.prepare_all(needs)
            decs.append(d)
            rec(kind="prepared", ms=ms, cuda_alloc_gb=torch.cuda.memory_allocated() / 1e9,
                cuda_reserved_gb=torch.cuda.memory_reserved() / 1e9, kv_gb=d.kv_bytes() / 1e9, **meminfo())
        # one arm per (model, full/narrow width): alternate between the widest of each model; nested cases also run the narrow ones
        arms = [(mi, li) for mi, (e, labels, _) in enumerate(mods) for li in (range(len(labels)) if args.case in ("pair3", "both") else [len(labels) - 1])]
        for i in pick[:2]:
            for mi, li in arms:
                decs[mi].get(li).generate(enc[i].cuda(), 4, [])
        if args.load:
            from jetson_runner import Load
            load = Load("gpu", 0.6, 2048, 4)
            rec(kind="load_running", attempts=load.start_attempts, **meminfo())
        v0, t0 = vmstat(), time.time()
        with MemPoller() as mp:
            for i in pick:
                for mi, li in arms:
                    for rep in (0, 1):
                        r = decs[mi].get(li).generate(enc[i].cuda(), 32, [])
                        rec_ = dict(kind="gen", model=mods[mi][1][li], params=mods[mi][1][li], rep=rep, prompt=plen[i], ttft_ms=r["ttft_ms"],
                                    ms_tok=r["ms_per_decode_token"], total_ms=r["total_ms"])
                        out.write(json.dumps(rec_) + "\n")
                if len(arms) > 1:
                    out.flush()
        v1 = vmstat()
        rec(kind="summary", case=args.case, load=args.load, seconds=time.time() - t0, mem_start_used_mb=mp.start_mb, mem_peak_used_mb=mp.peak_mb,
            cuda_peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9, cuda_peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
            swap_in_pages=v1["pswpin"] - v0["pswpin"], swap_out_pages=v1["pswpout"] - v0["pswpout"], **meminfo())
        if load is not None:
            rec(kind="load_ops", ops=load.ops(), alive=load.procs[0].poll() is None)
        rec(kind="done", ok=True)
    except Exception as ex:                                   # out of memory is an outcome, not a crash
        rec(kind="failed", error=type(ex).__name__, message=str(ex)[:300], **meminfo(), **vmstat())
        sys.exit(3)
    finally:
        if load is not None:
            load.stop()


if __name__ == "__main__":
    main()
