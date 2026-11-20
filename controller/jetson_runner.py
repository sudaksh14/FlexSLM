#!/usr/bin/env python3
"""Run the width controller (controller/policies.py) for real on the Jetson.

One MedNest export is loaded once; every width is a view of its weights (jetson/mednest_jetson/elastic.py)
with CUDA-graph decoders sharing one KV allocation (decode.py). A stratified request stream from a bundle
(controller/make_bundle.py) is served one request at a time; per request the policy picks a width, the
board serves it, and the policy observes the MEASURED latency and energy (VDD_IN rail integrated over the
request's own time window, net of the idle floor). Every policy sees the same stream and the same phases.

Phases disturb the device mid-stream, e.g. `15W:210,15W+gpu:210,15W:210`:
  MODE        nvpmodel mode to set (needs passwordless sudo nvpmodel), or `keep` (leave it as it is)
  +gpu / +cpu a co-running load for the phase: a CUDA matmul loop with --load-duty, or busy CPU processes
The controller only ever knows ONE profile (--profile, measured at 15 W without load), as a deployed
controller would: the open-loop `deadline` policy cannot see a disturbance, the `+fb` / queue policies
learn it from observed latency.

Scoring (--score):
  cluster  (default; works with random-weight exports, no checkpoint on the board) latency/energy are measured
           on the board with each request decoded to exactly the token count the trained model produced on
           the cluster at that width (EOS ignored); correctness = the cluster's answer at that width.
  device   the trained export is on the board: greedy decoding with EOS (max 32 tokens), the board's answer
           is scored, and agreement with the cluster answer is recorded.

Policies: fixed:<k>|max|min, deadline, deadline+fb, queue:<f>, queue-task:<f>  (queue budget = E(min) + f x
(E(max) - E(min)), energies predicted by the profile at the stream's median prompt; queues use feedback and
the deadline filter).

    python3 controller/jetson_runner.py --bundle controller/bundles/paper.qwen3_0p6b_nested.n630.s0.json \
        --elastic ~/mednest/models/mednest-random --profile ~/mednest-results/shape_only_full/widths_15W.jsonl \
        --phases 15W:210,15W+gpu:210,15W:210 --policies fixed:max fixed:min deadline deadline+fb queue-task:0.5
"""
from __future__ import annotations

import bisect
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "jetson"))
from mednest_jetson.cli import base_parser, cooldown, setup, torch_dtype  # noqa: E402

GPU_LOAD = r"""
import mmap, struct, sys, time, torch
size, duty, period = int(sys.argv[1]), float(sys.argv[2]), 0.05
f = open(sys.argv[3], "r+b")
m = mmap.mmap(f.fileno(), 8)
a = torch.randn(size, size, device="cuda", dtype=torch.float16) / size ** 0.5
n = 0
while True:
    t = time.monotonic()
    while time.monotonic() - t < period * duty:
        b = a @ a
        torch.cuda.synchronize()
        n += 1
        m[:8] = struct.pack("q", n)              # finished matmuls, read by the runner (energy attribution)
    time.sleep(period * (1 - duty))
"""
CPU_LOAD = "while True: pass"


class Load:
    """A co-running process for one phase (started/stopped around it)."""

    def __init__(self, kind, duty, size, n_cpu, tries=3, ready_s=60.0):
        self.kind, self.procs, self.counter = kind, [], None
        if kind == "gpu":
            # The load's CUDA context needs ~1 GB of the board's shared memory next to the resident model and
            # decoders; at start-up it can fail with an out-of-memory error after this process has moved on.
            # Wait until it has really finished matmuls, retry with the caching allocator emptied, and abort the
            # run if it never starts (a "loaded" phase without load would be silently wrong).
            import gc
            import tempfile
            import torch
            for attempt in range(1, tries + 1):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                fd, self.counter = tempfile.mkstemp(prefix="ctrl_load_",
                                                    dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
                os.write(fd, b"\0" * 8)
                os.close(fd)
                p = subprocess.Popen([sys.executable, "-c", GPU_LOAD, str(size), str(duty), self.counter])
                self.procs = [p]
                end = time.monotonic() + ready_s
                while time.monotonic() < end and p.poll() is None and self.ops() < 10:
                    time.sleep(0.2)
                if p.poll() is None and self.ops() >= 10:
                    self.start_attempts = attempt
                    break
                print(f"    [load] gpu load not running (attempt {attempt}/{tries}, exit {p.poll()})", flush=True)
                self.stop()
                time.sleep(5.0)
            else:
                raise RuntimeError(f"gpu load did not start after {tries} attempts")
        elif kind == "cpu":
            self.procs = [subprocess.Popen([sys.executable, "-c", CPU_LOAD]) for _ in range(n_cpu)]
            time.sleep(0.5)
        for p in self.procs:
            if p.poll() is not None:
                raise RuntimeError(f"{kind} load process exited with {p.returncode}")

    def ops(self):
        """Work units the load has finished so far (GPU: matmuls), or None (no counter)."""
        if self.counter is None:
            return None
        import struct
        with open(self.counter, "rb") as f:
            return struct.unpack("q", f.read(8))[0]

    def stop(self):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []
        if self.counter is not None:
            try:
                os.remove(self.counter)
            except OSError:
                pass
            self.counter = None

    def alive(self):
        return bool(self.procs) and all(p.poll() is None for p in self.procs)


def parse_phases(spec):
    out = []
    for part in spec.split(","):
        head, n = part.rsplit(":", 1)
        mode, _, load = head.partition("+")
        out.append((mode, load or None, int(n)))
    return out


def window_energy_j(trace, rail, t0, t1, wait_s=0.03):
    """Rail energy (J) between monotonic times t0 and t1, trapezoids clipped to the window."""
    if rail is None:
        return None
    end = time.monotonic() + wait_s
    while (not trace.t or trace.t[-1] < t1) and time.monotonic() < end:
        time.sleep(0.002)                            # let the poller take its first sample after t1
    t, p = trace.t, trace.rails.get(rail, [])
    n = min(len(t), len(p))
    i0 = max(0, bisect.bisect_left(t, t0, 0, n) - 1)
    i1 = min(n, bisect.bisect_right(t, t1, 0, n) + 1)
    if i1 - i0 < 2:
        return None
    e = 0.0
    for i in range(i0 + 1, i1):
        lo, hi = max(t[i - 1], t0), min(t[i], t1)
        if hi > lo:
            e += 0.5 * (p[i] + p[i - 1]) * (hi - lo)
    return e / 1000.0                                # mW * s -> J


def baseline_power_w(trace, rail, seconds, load=None):
    """Mean rail power (W) over the next `seconds` with no request running (None without rails), and the
    load's work rate over the same window (ops/s; None without a counter)."""
    if rail is None or seconds <= 0:
        return None, None
    c0 = load.ops() if load is not None else None
    t0 = time.monotonic()
    time.sleep(seconds)
    t1 = time.monotonic()
    c1 = load.ops() if load is not None else None
    e = window_energy_j(trace, rail, t0, t1)
    rate = (c1 - c0) / (t1 - t0) if c0 is not None and c1 is not None else None
    return (None if e is None else e / (t1 - t0)), rate


def make_policy(spec, prof, levels, bundle, budget_energy):
    import numpy as np
    from policies import DeadlinePolicy, EnergyQueuePolicy, Fixed
    n_pred = bundle["n_pred_tokens"]
    if spec.startswith("fixed:"):
        v = spec.split(":")[1]
        return Fixed(max(levels) if v == "max" else min(levels) if v == "min" else int(v))
    if spec in ("deadline", "deadline+fb"):
        return DeadlinePolicy(prof, levels, answer_tokens=n_pred, feedback=spec.endswith("+fb"))
    if spec.startswith("queue"):
        kind, f = spec.split(":")
        task = kind == "queue-task"
        lv = bundle["levels"]
        idx = [lv.index(l) for l in levels]
        acc = np.array(bundle["acc_task_cal"])[:, idx] if task else np.array(bundle["acc_cal"])[idx]
        lo, hi = budget_energy[min(levels)], budget_energy[max(levels)]
        B = lo + float(f) * (hi - lo)
        p = EnergyQueuePolicy(prof, levels, acc, budget_j=B, V=10.0, answer_tokens=n_pred, feedback=True,
                              task_aware=task)
        p.name = f"{kind}:{f}"
        return p
    raise ValueError(f"unknown policy {spec}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--elastic", default=None, help="MedNest export dir with level<k>/ subdirs (widest has weights)")
    p.add_argument("--random-from", default=None,
                   help="instead of --elastic: dir of the base model (config + tokenizer, e.g. ~/mednest/models/"
                        "Qwen3-0.6B); random seeded weights built in memory, widths of jetson/make_random_export.py; "
                        "nothing written to disk (shape-only, --score cluster only)")
    p.add_argument("--levels", type=int, nargs="*", default=None, help="default: every level in bundle and export")
    p.add_argument("--profile", required=True, help="bench_widths.py output the controller predicts from")
    p.add_argument("--phases", default="keep:210,keep+gpu:210,keep:210")
    p.add_argument("--policies", nargs="*",
                   default=["fixed:max", "fixed:min", "deadline", "deadline+fb", "queue:0.5", "queue-task:0.5"])
    p.add_argument("--score", default="cluster", choices=["cluster", "device"])
    p.add_argument("--urgent-frac", type=float, default=0.3)
    p.add_argument("--urgent-ms", default="auto", help="ms, or auto = median predicted latency of the widest width")
    p.add_argument("--routine-ms", default="auto", help="ms, or auto = 1.5 x p90 predicted latency of the widest")
    p.add_argument("--no-deadlines", action="store_true", help="budget-only run: no request has a deadline")
    p.add_argument("--load-duty", type=float, default=0.6)
    p.add_argument("--load-size", type=int, default=2048)
    p.add_argument("--load-cpus", type=int, default=4)
    p.add_argument("--idle-s", type=float, default=10.0)
    p.add_argument("--load-baseline-s", type=float, default=5.0,
                   help="s of rail sampling with the phase's load running and no request, at the start and end of "
                        "each loaded phase; that power (not idle) is subtracted from the phase's requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decode", default="graph", choices=["graph", "eager"])
    args = p.parse_args()

    import random
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    import data_io as D
    from mednest_jetson import elastic as el
    from mednest_jetson import workload as wl
    from mednest_jetson.decode import ElasticDecoders, max_len_for
    from nestbench import device
    from nestbench.power import PowerSampler, measure_idle
    from policies import Request
    from simulate import summarize

    if args.score == "cluster":
        os.environ["MEDNEST_FORCE_LEN"] = "1"        # exact cluster answer lengths, EOS ignored
    w = setup(args, "controller")
    bundle = json.load(open(args.bundle, encoding="utf8"))
    reqs = bundle["requests"]
    phases = parse_phases(args.phases)
    n_total = sum(n for _, _, n in phases)
    if n_total > len(reqs):
        sys.exit(f"phases need {n_total} requests, bundle has {len(reqs)}")
    reqs = reqs[:n_total]
    if (args.elastic is None) == (args.random_from is None):
        sys.exit("give exactly one of --elastic / --random-from")
    if args.random_from:
        if args.score != "cluster":
            sys.exit("--random-from has random weights: only --score cluster")
        import make_random_export as mre             # jetson/ is on sys.path (mednest_jetson import)
        widths = [dict(hidden_size=h, num_attention_heads=a, num_key_value_heads=k, intermediate_size=m)
                  for h, a, k, m in zip(mre.HIDDEN, mre.HEADS, mre.KV, mre.MLP)]
        avail = list(range(len(widths)))
        tok_dir = args.random_from
    else:
        dirs = el.level_dirs_of(args.elastic)
        avail = sorted(dirs)
        tok_dir = dirs[max(dirs)]
    levels = sorted(set(args.levels if args.levels is not None else avail) & set(avail) & set(bundle["levels"]))
    prof = D.load_profile(args.profile)
    missing = set(levels) - set(prof.levels)
    if missing:
        sys.exit(f"profile {args.profile} has no level(s) {sorted(missing)}")
    li = {l: bundle["levels"].index(l) for l in levels}
    dtype = torch_dtype(args.dtype)

    tok = AutoTokenizer.from_pretrained(tok_dir)
    enc = [wl.encode(tok, wl.prompt_for(r, "as_is"), "as_is", "cpu") for r in reqs]
    plen = [int(x.shape[1]) for x in enc]
    if args.score == "cluster":    # tokens the trained model generated at each width (+1 for its end token)
        n_new = [{l: max(1, min(32, len(tok(r["gen"][li[l]], add_special_tokens=False).input_ids) + 1))
                  for l in levels} for r in reqs]
    else:
        n_new = [{l: 32 for l in levels} for _ in reqs]

    # deadlines and queue budgets from the controller's own profile (never from this run's measurements)
    n_pred = bundle["n_pred_tokens"]
    top = max(levels)
    pred_top = np.array([prof.latency_ms(top, x, n_pred) for x in plen])
    urgent_ms = float(np.median(pred_top)) if args.urgent_ms == "auto" else float(args.urgent_ms)
    routine_ms = 1.5 * float(np.percentile(pred_top, 90)) if args.routine_ms == "auto" else float(args.routine_ms)
    rng = random.Random(args.seed)
    urgent = [rng.random() < args.urgent_frac for _ in reqs]
    deadlines = [None if args.no_deadlines else (urgent_ms if u else routine_ms) for u in urgent]
    med = float(np.median(plen))
    budget_energy = {l: float(prof.energy_j(l, prof.latency_ms(l, med, n_pred))) for l in levels}
    w.write({"kind": "setup", "bundle": os.path.basename(args.bundle), "model": bundle["model"], "levels": levels,
             "phases": phases, "policies": args.policies, "score": args.score, "urgent_ms": urgent_ms,
             "routine_ms": routine_ms, "urgent_frac": args.urgent_frac, "no_deadlines": args.no_deadlines,
             "budget_energy_pred_j": budget_energy, "profile": args.profile, "n_requests": n_total,
             "weights": args.elastic or f"random(seed {args.seed}) from {args.random_from}",
             "load": {"duty": args.load_duty, "size": args.load_size, "cpus": args.load_cpus}})
    print(f"[controller] {n_total} requests, levels {levels}, urgent {urgent_ms:.0f} ms, routine {routine_ms:.0f} ms, "
          f"score={args.score}", flush=True)

    if args.random_from:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(args.random_from)
        for k, v in widths[-1].items():
            if getattr(cfg, k) != v:
                sys.exit(f"widest width differs from {args.random_from} in {k}")
        torch.manual_seed(args.seed)
        with torch.device(args.device):
            full = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype).eval()
        full.generation_config = __import__("transformers").GenerationConfig.from_pretrained(args.random_from)
        e = el.ElasticModel.from_widths(full, widths)
    else:
        e = el.ElasticModel.load(dirs[top], [dirs[l] for l in sorted(dirs)], device=args.device, dtype=dtype)
    e.build_all()
    eos = wl.eos_ids(e.width(top), tok)
    decs = ElasticDecoders(e, max_len_for(plen, 32), use_graph=(args.decode == "graph"))
    t0 = time.perf_counter()
    decs.prepare_all([x + 32 for x in plen], levels)
    for i in range(min(args.warmup, len(enc))):      # untimed: kernels, allocator, clocks
        for l in levels:
            decs.get(l).generate(enc[i].to(args.device), 4, eos)
    w.write({"kind": "warmup", "ms": (time.perf_counter() - t0) * 1000.0})

    idle = measure_idle(source=args.power_source, seconds=args.idle_s, period_ms=args.power_period_ms)
    idle_w = idle.get("idle_power_w")
    w.write({"kind": "idle", **idle})
    start_mode = device.current_power_mode().get("name") if device.is_jetson() else None
    cur_mode = start_mode

    try:
        for spec in args.policies:
            pol = make_policy(spec, prof, levels, bundle, budget_energy)
            cooldown(args)
            recs, i, prev = [], 0, None
            with PowerSampler(source=args.power_source, period_ms=args.power_period_ms) as ps:
                rail = ps.trace.total_rail() if ps.trace.n() else None
                for mode, load_kind, n in phases:
                    if mode != "keep" and mode != cur_mode:
                        device.set_power_mode(mode, require=True)
                        cur_mode = mode
                    load = Load(load_kind, args.load_duty, args.load_size, args.load_cpus) if load_kind else None
                    # Loaded phase: the load's draw is taken out of each request's energy. Baseline window (load
                    # alone): P_load above idle and its work rate -> J per load op. A request is charged
                    #   gross - idle * dt - J_per_op * (load ops finished during the request)
                    # ("ops" attribution: the load slows while it shares the GPU, and clock drift changes ops and
                    # power together). Bounds kept per request: gross - P_load * dt (whole standalone draw
                    # removed: lower) and gross - idle * dt (nothing removed: upper).
                    base_w, j_per_op = idle_w, None
                    if load is not None:
                        lb, rate = baseline_power_w(ps.trace, rail, args.load_baseline_s, load)
                        if lb is not None:
                            base_w = lb
                            if rate and idle_w is not None:
                                j_per_op = max(0.0, lb - idle_w) / rate
                        w.write({"kind": "load_baseline", "policy": pol.name, "phase": f"{mode}+{load_kind}",
                                 "at": "start", "power_w": lb, "ops_per_s": rate, "j_per_op": j_per_op,
                                 "idle_power_w": idle_w, "seconds": args.load_baseline_s})
                    try:
                        for _ in range(n):
                            r = reqs[i]
                            task = D.SOURCES.index(r["source"]) if r["source"] in D.SOURCES else None
                            level = pol.choose(Request(prompt_tokens=plen[i], deadline_ms=deadlines[i], task=task))
                            sw = e.switch(level)["switch_ms"] if level != prev else 0.0
                            ids = enc[i].to(args.device)
                            ops0 = load.ops() if load is not None else None
                            ta = time.monotonic()
                            g = decs.get(level).generate(ids, n_new[i][level], eos)
                            tb = time.monotonic()
                            ops1 = load.ops() if load is not None else None
                            if load is not None and not load.alive():
                                raise RuntimeError(f"{load_kind} load exited during the phase (request {i})")
                            lat = g["total_ms"] + sw
                            gross = window_energy_j(ps.trace, rail, ta, tb)
                            en_idle = en_lo = load_ops = None
                            base_src = "idle"
                            if gross is not None and idle_w is not None:
                                en_idle = gross - idle_w * (tb - ta)
                                en_lo = gross - base_w * (tb - ta)
                                if load is None:
                                    en = en_idle
                                elif j_per_op is not None and ops0 is not None:
                                    load_ops = ops1 - ops0
                                    en, base_src = en_idle - j_per_op * load_ops, "load_ops"
                                else:                       # no counter (cpu load): whole standalone draw removed
                                    en, base_src = en_lo, "load_power"
                                en_src = "measured"
                            else:
                                en, en_src = float(prof.energy_j(level, lat)), "model"
                            pol.observe(level, lat, en)
                            gen = tok.decode(g.pop("new_ids"), skip_special_tokens=True)
                            cl_ok = r["correct"][li[level]]
                            if args.score == "cluster":
                                ok, agree = cl_ok, None
                            else:
                                ok = wl.score(r, gen, "as_is")["correct"]
                                agree = (wl.score(r, r["gen"][li[level]], "as_is").get("boxed")
                                         == wl.score(r, gen, "as_is").get("boxed"))
                            tag = ("urgent" if urgent[i] else "routine") if deadlines[i] is not None else "none"
                            tag += f"@{mode}" + (f"+{load_kind}" if load_kind else "")
                            pred = getattr(pol, "pred", None)
                            rec = {"kind": "request", "policy": pol.name, "i": i, "test_index": r["test_index"],
                                   "source": r["source"], "task": task, "phase": tag, "level": level,
                                   "deadline_ms": deadlines[i], "latency_ms": lat, "met": (deadlines[i] is None
                                   or lat <= deadlines[i]), "switch_ms": sw, "energy_j": en, "energy_src": en_src,
                                   "energy_gross_j": gross, "energy_net_idle_j": en_idle,
                                   "energy_net_loadpower_j": en_lo, "load_ops": load_ops, "j_per_load_op": j_per_op,
                                   "baseline_w": base_w, "baseline_src": base_src,
                                   "predicted_ms": float(prof.latency_ms(level, plen[i], n_pred)),
                                   "rho": pred.rho if pred is not None else None, "Q": getattr(pol, "Q", None),
                                   "correct": bool(ok), "cluster_correct": bool(cl_ok), "agree_cluster": agree,
                                   **{k: v for k, v in g.items() if k not in ("decoder",)}}
                            w.write(rec)
                            recs.append((task, bool(ok), lat, en, level, deadlines[i], tag))
                            prev = level
                            i += 1
                        if load is not None:         # drift check of the load's draw over the phase
                            lb, rate = baseline_power_w(ps.trace, rail, args.load_baseline_s, load)
                            w.write({"kind": "load_baseline", "policy": pol.name, "phase": f"{mode}+{load_kind}",
                                     "at": "end", "power_w": lb, "ops_per_s": rate,
                                     "j_per_op": (max(0.0, lb - idle_w) / rate if lb is not None and rate
                                                  and idle_w is not None else None),
                                     "idle_power_w": idle_w, "seconds": args.load_baseline_s})
                    finally:
                        if load is not None:
                            load.stop()
            s = summarize(pol.name, recs)
            s.update({"kind": "policy_summary", "power": ps.summary(), "idle_power_w": idle_w,
                      "energy_src": "measured" if idle_w is not None and rail is not None else "model"})
            w.write(s)
            print(f"  == {pol.name}: acc {s['macro_acc']:.1f}  J {s['energy_j']:.2f}  mean {s['mean_ms']:.0f} ms"
                  + (f"  miss {100 * s['miss_rate']:.1f}% {dict((k, round(100 * v, 1)) for k, v in s.get('miss_by_tag', {}).items())}"
                     if "miss_rate" in s else "") + f"  levels {s['levels']}", flush=True)
    finally:
        if start_mode is not None and cur_mode != start_mode:
            device.set_power_mode(start_mode, require=False)
        w.close()


if __name__ == "__main__":
    main()
