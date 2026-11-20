"""Energy-matched comparisons on a board run (controller_*.jsonl from jetson_runner.py).

    python controller/board_compare.py <controller.jsonl> [--pairs queue-task:0.5,queue:0.5 deadline+fb,deadline]

1. Time-sharing reference: the board's fixed:min and fixed:max runs define the line a deployment gets by mixing
   the narrowest and the widest width in proportion (accuracy and energy both linear in the mix). Each policy is
   compared with that line at the policy's own measured energy (paired bootstrap over the same stream: the mix is
   scored per request as w * correct_max + (1 - w) * correct_min).
2. Paired accuracy differences (and deadline-miss differences) between named policies.
"""
from __future__ import annotations

import argparse

import numpy as np

from analyze_board import load


def macro(correct, task, w=None):
    """Mean over test sets of per-set accuracy; `correct` may be fractional (a mix)."""
    return 100.0 * float(np.mean([correct[task == s].mean() for s in np.unique(task)]))


def boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pairs", nargs="*", default=["queue-task:0.5,queue:0.5", "deadline+fb,deadline"])
    args = ap.parse_args()
    _, reqs, _, _ = load(args.path)
    arr = {}
    for name, rs in reqs.items():
        rs = sorted(rs, key=lambda r: r["i"])
        arr[name] = {"ok": np.array([r["correct"] for r in rs], float), "task": np.array([r["task"] for r in rs]),
                     "J": np.array([r["energy_j"] for r in rs]),
                     "miss": np.array([0.0 if r["met"] else 1.0 for r in rs]),
                     "has_dl": any(r["deadline_ms"] is not None for r in rs)}
    lo = next(k for k in arr if k.startswith("fixed:") and k != max(
        (k for k in arr if k.startswith("fixed:")), key=lambda k: arr[k]["J"].mean()))
    hi = max((k for k in arr if k.startswith("fixed:")), key=lambda k: arr[k]["J"].mean())
    Jlo, Jhi = arr[lo]["J"].mean(), arr[hi]["J"].mean()
    task = arr[hi]["task"]
    n = len(task)
    print(f"time-sharing line: {lo} ({macro(arr[lo]['ok'], task):.1f}, {Jlo:.2f} J) -- {hi} "
          f"({macro(arr[hi]['ok'], task):.1f}, {Jhi:.2f} J)")
    for name, a in arr.items():
        if name in (lo, hi):
            continue
        J = a["J"].mean()
        w = min(1.0, max(0.0, (J - Jlo) / (Jhi - Jlo)))
        mix = w * arr[hi]["ok"] + (1 - w) * arr[lo]["ok"]
        d = macro(a["ok"], task) - macro(mix, task)
        ci = boot(lambda i: macro(a["ok"][i], task[i]) - macro(mix[i], task[i]), n)
        print(f"  {name:16s} acc {macro(a['ok'], task):5.1f} at {J:.2f} J; mix at that energy "
              f"{macro(mix, task):5.1f} (w_max {w:.2f}); policy - mix {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    for pair in args.pairs:
        x, y = pair.split(",")
        if x not in arr or y not in arr:
            continue
        a, b = arr[x], arr[y]
        d = macro(a["ok"], task) - macro(b["ok"], task)
        ci = boot(lambda i: macro(a["ok"][i], task[i]) - macro(b["ok"][i], task[i]), n)
        line = f"  {x} - {y}: acc {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]  J {a['J'].mean() - b['J'].mean():+.2f}"
        if a["has_dl"] and b["has_dl"]:
            dm = 100 * (a["miss"].mean() - b["miss"].mean())
            cm = boot(lambda i: 100 * (a["miss"][i].mean() - b["miss"][i].mean()), n)
            line += f"  misses {dm:+.1f} pts [{cm[0]:+.1f}, {cm[1]:+.1f}]"
        print(line)


if __name__ == "__main__":
    main()
