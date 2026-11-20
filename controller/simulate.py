"""Trace-driven simulation of the runtime controller.

    python controller/simulate.py budget   [--answer eos|forced32] [--n 4000]
    python controller/simulate.py thermal  [--answer eos|forced32] [--n 1200]

What is real and what is synthetic:
- REAL: whether each question is answered correctly at each width (M3 eval outputs; scored only on the
  evaluation half, the controller's accuracy tables come from the calibration half); latency and power per
  width as a function of prompt length (Jetson Orin Nano 15 W profile); latency noise (the empirical
  observed/predicted ratios of the Jetson run).
- SYNTHETIC: the request stream (uniform over the evaluation half, seeded), deadlines, and the device
  phases of `thermal` (a latency slowdown factor standing in for throttling / co-running load; power held
  constant). Validate on the board before reporting anything as a device result.

Scenarios:
- budget:  long-run energy budget, no deadlines. Fixed widths vs EnergyQueue (task-agnostic, task-aware),
           against the convex-hull mix (the best any width-only controller can do on average).
- thermal: urgent/routine deadlines; a slowdown phase in the middle. Fixed, open-loop deadline rule,
           deadline rule with latency feedback, and feedback + energy queue.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

import data_io as D
from offline import costs, hull_value, upper_hull
from policies import DeadlinePolicy, EnergyQueuePolicy, Fixed, Policy, Request


def task_table(raw: np.ndarray, glob: np.ndarray, corr: D.Correctness, cal: np.ndarray, mode: str,
               kappa: float = 500.0) -> np.ndarray:
    """Per-task accuracy tables for the task-aware queue, from the calibration half.
    raw:    per-set accuracy as measured (noisy for small sets: Bullets has 142 calibration questions).
    shrunk: (n_s * raw_s + kappa * global) / (n_s + kappa).
    scaled: raw_s(L0) + beta_s * (global - global(L0)), beta_s >= 0 by least squares: one sensitivity per task."""
    if mode == "raw":
        return raw
    n = np.array([(cal & (corr.source == s)).sum() for s in range(raw.shape[0])], float)[:, None]
    if mode == "shrunk":
        return (n * raw + kappa * glob[None]) / (n + kappa)
    g = glob - glob[0]
    out = []
    for r in raw:
        beta = max(0.0, float(np.dot(r - r[0], g) / np.dot(g, g)))
        out.append(r[0] + beta * g)
    return np.array(out)


class World:
    def __init__(self, answer: str = "eos", seed: int = 0, model: str = D.M3, task_mode: str = "shrunk"):
        self.corr, self.prof = D.load_correctness(model), D.load_profile()
        self.L = self.corr.levels
        self.cal = D.split_mask(len(self.corr.source), seed)
        self.ptok = D.prompt_tokens(self.corr, self.prof)
        self.ntok = D.answer_tokens(self.corr, answer)
        self.lat, self.en = costs(self.corr, self.prof, answer)
        # controller-side statistics: calibration half only
        self.acc_cal = self.corr.macro(self.cal)                 # [L]
        self.acc_task_raw = self.corr.per_source(self.cal)       # [S, L]
        self.acc_task_cal = task_table(self.acc_task_raw, self.acc_cal, self.corr, self.cal, task_mode)
        self.n_pred = float(np.percentile(self.ntok[self.cal], 90))
        self.E_cal = self.en[self.cal].mean(0)
        self.eval_idx = np.flatnonzero(~self.cal)

    def stream(self, n: Optional[int], seed: int) -> np.ndarray:
        """The evaluation half in a seeded random order (each question once); n > half repeats it."""
        rng = np.random.default_rng(seed)
        n = n or len(self.eval_idx)
        reps = -(-n // len(self.eval_idx))
        return np.concatenate([rng.permutation(self.eval_idx) for _ in range(reps)])[:n]


def run(world: World, policy: Policy, qs: np.ndarray, deadlines: Optional[Sequence[Optional[float]]] = None,
        slowdown: Optional[Sequence[float]] = None, seed: int = 0, tags: Optional[Sequence[str]] = None) -> dict:
    rng = np.random.default_rng(seed)       # same noise draws for every policy
    noise = rng.choice(world.prof.ratio_samples, size=len(qs))
    j_of = {l: j for j, l in enumerate(world.L)}
    recs = []
    prev = None
    for i, q in enumerate(qs):
        dl = deadlines[i] if deadlines is not None else None
        req = Request(prompt_tokens=int(world.ptok[q]), deadline_ms=dl, task=int(world.corr.source[q]))
        lvl = policy.choose(req)
        j = j_of[lvl]
        s = slowdown[i] if slowdown is not None else 1.0
        lat = float(world.lat[q, j] * noise[i] * s)
        lat += world.prof.switch_view_ms if (prev is not None and prev != lvl) else 0.0
        en = float(world.prof.energy_j(lvl, lat))
        policy.observe(lvl, lat, en)
        recs.append((int(world.corr.source[q]), bool(world.corr.C[q, j]), lat, en, lvl, dl,
                     tags[i] if tags is not None else None))
        prev = lvl
    out = summarize(policy.name, recs)
    out["_correct"] = np.array([r[1] for r in recs])
    out["_source"] = np.array([r[0] for r in recs])
    return out


def macro_of(correct: np.ndarray, source: np.ndarray, idx: np.ndarray) -> float:
    c, s = correct[idx], source[idx]
    return 100 * float(np.mean([c[s == k].mean() for k in np.unique(s)]))


def paired_ci(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Macro-accuracy difference a - b on the same request stream, 95% bootstrap CI over requests."""
    rng = np.random.default_rng(seed)
    n = len(a["_correct"])
    d = [macro_of(a["_correct"], a["_source"], i) - macro_of(b["_correct"], b["_source"], i)
         for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    full = macro_of(a["_correct"], a["_source"], np.arange(n)) - macro_of(b["_correct"], b["_source"], np.arange(n))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(full), float(lo), float(hi)


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def summarize(name: str, recs: List[Tuple]) -> dict:
    by = defaultdict(list)
    for s, ok, *_ in recs:
        by[s].append(ok)
    lat = np.array([r[2] for r in recs])
    with_dl = [r for r in recs if r[5] is not None]
    out = {"policy": name, "n": len(recs), "macro_acc": 100 * float(np.mean([np.mean(v) for v in by.values()])),
           "energy_j": float(np.mean([r[3] for r in recs])), "mean_ms": float(lat.mean()),
           "p90_ms": float(np.percentile(lat, 90)),
           "levels": dict(sorted(Counter(r[4] for r in recs).items())),
           "switches": int(sum(a[4] != b[4] for a, b in zip(recs, recs[1:])))}
    if with_dl:
        out["miss_rate"] = float(np.mean([r[2] > r[5] for r in with_dl]))
        tags = sorted({r[6] for r in with_dl if r[6] is not None})
        out["miss_by_tag"] = {t: float(np.mean([r[2] > r[5] for r in with_dl if r[6] == t])) for t in tags}
    return out


def fmt(r: dict) -> str:
    s = f"  {r['policy']:22s} acc {r['macro_acc']:5.1f}  J {r['energy_j']:5.2f}  mean {r['mean_ms']:5.0f} ms"
    if "miss_rate" in r:
        s += f"  miss {100*r['miss_rate']:5.1f}%"
        s += "".join(f" [{t} {100*v:4.1f}%]" for t, v in r.get("miss_by_tag", {}).items())
    return s + f"  switches {r['switches']:4d}  levels {r['levels']}"


def scenario_budget(w: World, n: Optional[int], seed: int, V: float) -> dict:
    qs = w.stream(n, seed)
    print(f"== budget scenario: {len(qs)} requests (evaluation half), answer tokens p90 {w.n_pred:.0f}, V {V}")
    fixed = {l: run(w, Fixed(l), qs, seed=seed) for l in w.L}
    for r in fixed.values():
        print(fmt(r))
    hull = upper_hull(list(w.E_cal), list(w.acc_cal))
    acc_ev = w.corr.macro(~w.cal)
    out = {"V": V, "fixed": [strip(r) for r in fixed.values()], "budgets": []}
    for B in np.linspace(w.E_cal[0], w.E_cal[-1], 7)[1:-1]:
        q = run(w, EnergyQueuePolicy(w.prof, w.L, w.acc_cal, budget_j=float(B), V=V, answer_tokens=w.n_pred,
                                     feedback=False), qs, seed=seed)
        qt = run(w, EnergyQueuePolicy(w.prof, w.L, w.acc_task_cal, budget_j=float(B), V=V, answer_tokens=w.n_pred,
                                      feedback=False, task_aware=True), qs, seed=seed)
        # best fixed width that stays within the budget (what a dense deployment would ship)
        fl = max([l for l in w.L if fixed[l]["energy_j"] <= B], default=w.L[0])
        ref = hull_value(list(w.E_cal), list(acc_ev), hull, B)
        d_task = paired_ci(qt, q)
        d_fixed = paired_ci(qt, fixed[fl])
        print(fmt(q)); print(fmt(qt))
        print(f"  {'hull mix (reference)':22s} acc {ref:5.1f}  J {B:5.2f}   best fixed within budget: L{fl} "
              f"{fixed[fl]['macro_acc']:.1f}")
        print(f"    task-aware - agnostic {d_task[0]:+.2f} [{d_task[1]:+.2f}, {d_task[2]:+.2f}]   "
              f"task-aware - fixed L{fl} {d_fixed[0]:+.2f} [{d_fixed[1]:+.2f}, {d_fixed[2]:+.2f}]")
        out["budgets"].append({"budget_j": float(B), "queue": strip(q), "queue_task": strip(qt), "hull_ref": ref,
                               "best_fixed": fl, "task_minus_agnostic": d_task, "task_minus_fixed": d_fixed})
    return out


def scenario_thermal(w: World, n: Optional[int], seed: int, slow: float, urgent_frac: float = 0.3,
                     urgent_level: int = 8, urgent_scale: float = 1.0) -> dict:
    """Urgent deadline = urgent_scale x the median predicted latency of `urgent_level`: met by that width in the
    nominal phase, so under the slowdown only a narrower width can meet it."""
    qs = w.stream(n, seed)
    n = len(qs)
    rng = np.random.default_rng(seed + 1)
    urgent = rng.random(n) < urgent_frac
    pred = {l: w.prof.latency_ms(l, w.ptok[qs], w.n_pred) for l in w.L}
    urgent_ms = urgent_scale * float(np.median(pred[urgent_level]))
    routine_ms = 1.2 * slow * float(np.percentile(pred[max(w.L)], 90))
    dls = [urgent_ms if u else routine_ms for u in urgent]
    third = n // 3
    slowdown = [slow if third <= i < 2 * third else 1.0 for i in range(n)]
    tags = [("urgent" if u else "routine") + ("-slow" if sl > 1 else "") for u, sl in zip(urgent, slowdown)]
    print(f"== thermal scenario: {n} requests, {urgent_frac:.0%} urgent ({urgent_ms:.0f} ms), routine "
          f"{routine_ms:.0f} ms, latency x{slow} in the middle third")
    B = float(np.mean([w.E_cal[0], w.E_cal[-1]]))
    pols = [Fixed(max(w.L)), Fixed(min(w.L)),
            DeadlinePolicy(w.prof, w.L, answer_tokens=w.n_pred, feedback=False),
            DeadlinePolicy(w.prof, w.L, answer_tokens=w.n_pred, feedback=True),
            EnergyQueuePolicy(w.prof, w.L, w.acc_task_cal, budget_j=B, V=10.0, answer_tokens=w.n_pred,
                              feedback=True, task_aware=True)]
    out = {"urgent_ms": urgent_ms, "routine_ms": routine_ms, "slowdown": slow, "results": []}
    res = []
    for p in pols:
        r = run(w, p, qs, dls, slowdown, seed=seed, tags=tags)
        res.append(r)
        out["results"].append(strip(r))
        print(fmt(r))
    d = paired_ci(res[3], res[2])
    print(f"  deadline+fb - deadline: acc {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]")
    out["fb_minus_openloop"] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["budget", "thermal"])
    ap.add_argument("--answer", default="eos", choices=["eos", "forced32"])
    ap.add_argument("--n", type=int, default=None, help="requests (default: the evaluation half, once)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--V", type=float, default=10.0)
    ap.add_argument("--slowdown", type=float, default=1.3)
    ap.add_argument("--urgent-level", type=int, default=8)
    ap.add_argument("--urgent-scale", type=float, default=1.0)
    ap.add_argument("--task-table", default="shrunk", choices=["raw", "shrunk", "scaled"])
    ap.add_argument("--model", default=D.M3, help="samples name in data/samples (A2: D.A2G, levels 0/4/8)")
    args = ap.parse_args()
    w = World(args.answer, args.seed, model=args.model, task_mode=args.task_table)
    if args.scenario == "budget":
        res = scenario_budget(w, args.n, args.seed, args.V)
    else:
        res = scenario_thermal(w, args.n, args.seed, args.slowdown, urgent_level=args.urgent_level,
                               urgent_scale=args.urgent_scale)
    os.makedirs(os.path.join(D.HERE, "out"), exist_ok=True)
    out = os.path.join(D.HERE, "out", f"sim.{args.scenario}.{args.answer}.{args.task_table}"
                       + ("" if args.model == D.M3 else "." + D.short_model(args.model)) + ".json")
    json.dump(res, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
