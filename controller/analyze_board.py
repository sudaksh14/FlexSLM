"""Tables from a jetson_runner.py output (controller*.jsonl), anywhere with numpy.

    python controller/analyze_board.py <controller.jsonl> [--ref fixed:L8] [--out controller/out/board.json]

Per policy: macro accuracy, mean net energy per answer (measured), mean / p90 latency, deadline misses per
phase, widths used, switches; paired 95% bootstrap CI of the accuracy difference to --ref (same stream).
Per width: measured vs profile-predicted latency by phase (how far the disturbance moved the device, and how
well the feedback factor rho tracked it at the end of each phase).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from simulate import paired_ci, summarize


def load(path):
    setup, reqs, summ = None, defaultdict(list), {}
    for l in open(path, encoding="utf8"):
        r = json.loads(l)
        k = r.get("kind")
        if k == "setup":
            setup = r
        elif k == "request":
            reqs[r["policy"]].append(r)
        elif k == "policy_summary":
            summ[r["policy"]] = r
    return setup, reqs, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default=None, help="policy name for the paired comparison (default: the first)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup, reqs, _ = load(args.path)
    if not reqs:
        raise SystemExit("no request records")
    print(f"phases {setup['phases']}  score={setup['score']}  urgent {setup['urgent_ms']:.0f} ms  "
          f"routine {setup['routine_ms']:.0f} ms")
    res = {}
    for name, rs in reqs.items():
        tup = [(r["task"], r["correct"], r["latency_ms"], r["energy_j"], r["level"], r["deadline_ms"], r["phase"])
               for r in rs]
        s = summarize(name, tup)
        s["_correct"], s["_source"] = np.array([r["correct"] for r in rs]), np.array([r["task"] for r in rs])
        s["energy_src"] = sorted({r["energy_src"] for r in rs})
        ag = [r["agree_cluster"] for r in rs if r.get("agree_cluster") is not None]
        s["agree_cluster"] = float(np.mean(ag)) if ag else None
        res[name] = s
    ref = args.ref or next(iter(res))
    print(f"{'policy':18s} {'acc':>5s} {'J/ans':>6s} {'mean':>5s} {'p90':>5s} {'miss':>6s}  diff vs {ref} [95% CI]")
    for name, s in res.items():
        d = paired_ci(s, res[ref]) if name != ref else (0.0, 0.0, 0.0)
        s["diff_vs_ref"] = d
        miss = f"{100 * s['miss_rate']:5.1f}%" if "miss_rate" in s else "   -  "
        print(f"{name:18s} {s['macro_acc']:5.1f} {s['energy_j']:6.2f} {s['mean_ms']:5.0f} {s['p90_ms']:5.0f} {miss}  "
              f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]  levels {s['levels']}")
        if s.get("miss_by_tag"):
            print(" " * 20 + "  ".join(f"{t} {100 * v:.1f}%" for t, v in s["miss_by_tag"].items()))
        if s["agree_cluster"] is not None:
            print(" " * 20 + f"agreement with cluster answers {100 * s['agree_cluster']:.1f}%")
    # device drift: measured / predicted latency per phase (all policies pooled)
    by = defaultdict(list)
    for rs in reqs.values():
        for r in rs:
            by[r["phase"].split("@")[-1]].append(r["latency_ms"] / r["predicted_ms"])
    print("measured / predicted latency by device phase:",
          {k: round(float(np.median(v)), 3) for k, v in by.items()})
    for name, rs in reqs.items():
        if rs[0].get("rho") is not None:
            ends = []          # (phase, rho at its last request), one entry per consecutive segment
            for r in rs:
                ph = r["phase"].split("@")[-1]
                if ends and ends[-1][0] == ph:
                    ends[-1] = (ph, round(r["rho"], 3))
                else:
                    ends.append((ph, round(r["rho"], 3)))
            print(f"  rho at the end of each phase ({name}): {ends}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                                   "board." + os.path.basename(args.path).replace(".jsonl", ".json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"setup": setup, "policies": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                            for k, v in res.items()}}, open(out, "w"), indent=1, default=float)
    print("->", out)


if __name__ == "__main__":
    main()
