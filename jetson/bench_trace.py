#!/usr/bin/env python3
"""Budget trace: does picking the width per request pay off when the budget changes?

A seeded stream of test questions is served one at a time. Each request is URGENT (deadline
--urgent-ms, e.g. bedside triage) with probability --urgent-frac, otherwise ROUTINE
(--routine-ms). The device's power mode changes during the stream (--phases, e.g. battery
saving: 15W -> 7W -> 15W), which changes every width's latency. Policies, each run for real on the
same stream:

  fixed:<level>     always the same width (the largest one = a conventional single model)
  elastic           MedNest: per request, the widest width whose predicted latency in the current
                    power mode meets the deadline (else the narrowest); switching = a view, free
  reload            the same choices, but every width is a separate model that must be loaded
                    from disk when the choice changes (one model resident); load time counts

Predicted latency per (power mode, width) comes from bench_widths.py answer records:
TTFT = a + b * prompt_tokens (least squares) plus the median decode ms/token times the p90 answer
length. Reported per policy: deadline-miss rate (all / urgent), accuracy (scored as on the
cluster), mean latency, energy per answer, switches.

    python3 jetson/bench_trace.py --elastic models/mednest --levels 0 4 8 \
        --profile 15W=~/mednest-results/widths_15W.jsonl --profile 7W=~/mednest-results/widths_7W.jsonl \
        --phases 15W:50,7W:50,15W:40 --policies fixed:8 fixed:0 elastic reload
"""
import gc
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mednest_jetson.cli import base_parser, cooldown, setup, torch_dtype  # noqa: E402


def fit_profile(path: str, levels):
    """{level: (a_ms, b_ms_per_prompt_token, decode_ms_per_token, n_new_p90)} from widths records."""
    from nestbench.results import load_results, stat_block
    _, rows = load_results(path)
    out = {}
    for l in levels:
        rs = [r for r in rows if r.get("kind") == "answer" and r.get("target") == "mednest" and r.get("level") == l]
        if len(rs) < 5:
            raise ValueError(f"{path}: fewer than 5 answers for level {l}")
        xs, ys = [r["prompt_tokens"] for r in rs], [r["ttft_ms"] for r in rs]
        n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx else 0.0
        a = my - b * mx
        dec = stat_block([r["ms_per_decode_token"] for r in rs if r["ms_per_decode_token"]])["median"]
        p90 = stat_block([r["n_new"] for r in rs])["p90"]
        out[l] = (a, b, dec, p90)
    return out


def predict(prof, level, prompt_tokens):
    a, b, dec, p90 = prof[level]
    return a + b * prompt_tokens + dec * max(0.0, p90 - 1)


def make_stream(rows, n, urgent_frac, seed):
    rng = random.Random(seed)
    pick = [rows[rng.randrange(len(rows))] for _ in range(n)]
    return [(r, rng.random() < urgent_frac) for r in pick]


def main():
    p = base_parser(__doc__)
    p.add_argument("--elastic", required=True)
    p.add_argument("--levels", type=int, nargs="*", default=None)
    p.add_argument("--profile", action="append", required=True, help="MODE=widths.jsonl (one per power mode)")
    p.add_argument("--phases", default="15W:50,7W:50,15W:40", help="MODE:N_REQUESTS,...")
    p.add_argument("--policies", nargs="*", default=["fixed:max", "fixed:min", "elastic", "reload"])
    p.add_argument("--urgent-frac", type=float, default=0.3)
    p.add_argument("--urgent-ms", default="auto",
                   help="ms, or auto = predicted latency of the middle width in the first phase's mode "
                        "(median over the subset's prompt lengths): the full width cannot meet it")
    p.add_argument("--routine-ms", default="auto",
                   help="ms, or auto = 1.2 x the full width's predicted latency in the slowest mode")
    p.add_argument("--subset", default=os.path.join(os.path.dirname(__file__), "questions", "subset.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decode", default="graph", choices=["graph", "eager"], help="as in bench_widths.py (use the same)")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mednest_jetson import elastic as el
    from mednest_jetson import workload as wl
    from mednest_jetson.decode import DenseDecoders, ElasticDecoders, max_len_for
    from nestbench import device
    from nestbench.power import PowerSampler
    from nestbench.results import stat_block

    w = setup(args, "trace")
    dtype = torch_dtype(args.dtype)
    dirs = el.level_dirs_of(args.elastic)
    levels = sorted(args.levels if args.levels is not None else dirs)
    profiles = {}
    for spec in args.profile:
        mode, _, path = spec.partition("=")
        profiles[mode] = fit_profile(os.path.expanduser(path), levels)
    phases = [(m, int(n)) for m, n in (x.split(":") for x in args.phases.split(","))]
    missing = {m for m, _ in phases} - set(profiles)
    if missing:
        sys.exit(f"no --profile for power mode(s) {sorted(missing)}")
    rows = wl.load_subset(args.subset)
    from transformers import AutoTokenizer as _AT
    _tok = _AT.from_pretrained(dirs[max(dirs)])
    plens = sorted(wl.encode(_tok, wl.prompt_for(r, "as_is"), "as_is", "cpu").shape[1] for r in rows)
    med_len = plens[len(plens) // 2]
    if args.urgent_ms == "auto":
        mid = levels[len(levels) // 2]
        args.urgent_ms = predict(profiles[phases[0][0]], mid, med_len)
    if args.routine_ms == "auto":
        args.routine_ms = 1.2 * max(predict(profiles[m], max(levels), med_len) for m in profiles)
    args.urgent_ms, args.routine_ms = float(args.urgent_ms), float(args.routine_ms)
    print(f"[trace] deadlines: urgent {args.urgent_ms:.0f} ms, routine {args.routine_ms:.0f} ms "
          f"(median prompt {med_len} tokens)", flush=True)
    trace_len = max_len_for(plens, 32)
    stream = make_stream(rows, sum(n for _, n in phases), args.urgent_frac, args.seed)
    w.write({"kind": "setup", "levels": levels, "phases": phases, "profiles": profiles,
             "urgent_ms": args.urgent_ms, "routine_ms": args.routine_ms, "n_requests": len(stream)})
    e = el.ElasticModel.load(dirs[max(dirs)], [dirs[l] for l in sorted(dirs)], device=args.device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(dirs[max(dirs)])
    decs = ElasticDecoders(e, trace_len) if args.decode == "graph" else None

    def choose(policy, mode, prompt_tokens, deadline):
        if policy.startswith("fixed:"):
            v = policy.split(":")[1]
            return max(levels) if v == "max" else min(levels) if v == "min" else int(v)
        fits = [l for l in levels if predict(profiles[mode], l, prompt_tokens) <= deadline]
        return max(fits) if fits else min(levels)

    for policy in args.policies:
        cooldown(args)
        recs, i, resident, resident_level, resident_dec = [], 0, None, None, None
        with PowerSampler(source=args.power_source, period_ms=args.power_period_ms) as ps:
            for mode, n in phases:
                if args.power_mode is not None or len(profiles) > 1:
                    device.set_power_mode(mode, require=True)
                for _ in range(n):
                    row, urgent = stream[i]
                    i += 1
                    deadline = args.urgent_ms if urgent else args.routine_ms
                    ids = wl.encode(tok, wl.prompt_for(row, "as_is"), "as_is", args.device)
                    level = choose(policy, mode, ids.shape[1], deadline)
                    switch_ms = 0.0
                    if policy == "reload":
                        if resident_level != level:
                            resident = resident_dec = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            resident = AutoModelForCausalLM.from_pretrained(dirs[level], torch_dtype=dtype).to(args.device).eval()
                            if args.decode == "graph":   # a freshly loaded model needs its own cache + graph
                                resident_dec = DenseDecoders(resident, trace_len)
                            torch.cuda.synchronize()
                            switch_ms = (time.perf_counter() - t0) * 1000.0
                            resident_level = level
                        model, dec = resident, resident_dec
                    else:
                        switch_ms = e.switch(level)["switch_ms"]
                        model, dec = e.width(level), (decs.get(level) if decs is not None else None)
                    eos = wl.eos_ids(model, tok)
                    r = dec.generate(ids, 32, eos) if dec is not None else wl.timed_greedy(model, ids, 32, eos)
                    gen = tok.decode(r.pop("new_ids"), skip_special_tokens=True)
                    latency = r["total_ms"] + switch_ms
                    rec = {"kind": "request", "policy": policy, "i": i, "power_mode": mode, "urgent": urgent,
                           "deadline_ms": deadline, "level": level, "switch_ms": switch_ms, "latency_ms": latency,
                           "predicted_ms": predict(profiles[mode], level, ids.shape[1]),
                           "met": latency <= deadline, "test_index": row.get("test_index"), "source": row["source"],
                           **wl.score(row, gen, "as_is"), **r}
                    w.write(rec)
                    recs.append(rec)
        pw = ps.summary()
        urg = [r for r in recs if r["urgent"]]
        summ = {"kind": "policy_summary", "policy": policy, "n": len(recs),
                "miss_rate": 1 - sum(r["met"] for r in recs) / len(recs),
                "miss_rate_urgent": (1 - sum(r["met"] for r in urg) / len(urg)) if urg else None,
                "accuracy": 100.0 * sum(r["correct"] for r in recs) / len(recs),
                "accuracy_by_source": wl.accuracy_by_source(recs),
                "latency_ms": stat_block([r["latency_ms"] for r in recs]),
                "switches": sum(1 for a, b in zip(recs, recs[1:]) if a["level"] != b["level"]),
                "switch_ms_total": sum(r["switch_ms"] for r in recs),
                "levels_used": {l: sum(r["level"] == l for r in recs) for l in levels},
                "energy_j_per_answer": (pw["energy_j"] / len(recs)) if pw.get("energy_j") is not None else None,
                "power": pw}
        w.write(summ)
        print(f"  == {policy}: miss {summ['miss_rate']:.1%} (urgent {summ['miss_rate_urgent'] or 0:.1%}), "
              f"acc {summ['accuracy']:.1f}, median latency {summ['latency_ms']['median']:.0f} ms, "
              f"switches {summ['switches']} ({summ['switch_ms_total']:.0f} ms)", flush=True)
        resident = None
        gc.collect()
        torch.cuda.empty_cache()
    w.close()


if __name__ == "__main__":
    main()
