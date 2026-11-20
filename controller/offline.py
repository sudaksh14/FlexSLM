"""Offline analysis: which widths are worth selecting, and how much a smarter controller could gain.

    python controller/offline.py [--answer eos|forced32] [--model <samples prefix>]

Writes controller/out/offline.<model>.<answer>.json and prints the tables. Numbers:
- cost per width = mean energy (J/answer) and mean latency (ms) over all 12,797 test questions, from the
  Jetson 15 W profile (data_io.Profile) at each question's own prompt length.
- Pareto set: widths no other width beats on both cost and accuracy.
- Upper convex hull: widths a controller under an AVERAGE budget would ever use (any point between two hull
  widths is reached by time-sharing them; widths under the hull are beaten by such a mix).
- Oracle (upper bound for per-question routing): for each question the cheapest width that answers it, if any.
- Task-aware static: per test set the width chosen on a calibration half, evaluated on the other half.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Sequence

import numpy as np

import data_io as D


def pareto(cost: Sequence[float], acc: Sequence[float]) -> List[int]:
    idx = sorted(range(len(cost)), key=lambda i: (cost[i], -acc[i]))
    out, best = [], -np.inf
    for i in idx:
        if acc[i] > best + 1e-9:
            out.append(i)
            best = acc[i]
    return out


def upper_hull(cost: Sequence[float], acc: Sequence[float]) -> List[int]:
    """Indices on the upper-left concave frontier (monotone chain over the Pareto set)."""
    pts = pareto(cost, acc)
    hull: List[int] = []
    for i in pts:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # keep b only if it lies strictly above the segment a -> i
            cross = (cost[b] - cost[a]) * (acc[i] - acc[a]) - (acc[b] - acc[a]) * (cost[i] - cost[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def hull_value(cost, acc, hull, budget: float) -> float:
    """Best average accuracy at an average cost <= budget by time-sharing two adjacent hull widths."""
    hc, ha = [cost[i] for i in hull], [acc[i] for i in hull]
    if budget <= hc[0]:
        return ha[0]
    if budget >= hc[-1]:
        return ha[-1]
    return float(np.interp(budget, hc, ha))


def costs(corr: D.Correctness, prof: D.Profile, answer: str):
    p = D.prompt_tokens(corr, prof)
    n = D.answer_tokens(corr, answer)
    lat = np.stack([prof.latency_ms(k, p, n[:, j]) for j, k in enumerate(corr.levels)], 1)   # [Q, L]
    en = np.stack([prof.energy_j(k, lat[:, j]) for j, k in enumerate(corr.levels)], 1)
    return lat, en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=D.M3)
    ap.add_argument("--answer", default="eos", choices=["eos", "forced32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corr, prof = D.load_correctness(args.model), D.load_profile()
    L = corr.levels
    lat, en = costs(corr, prof, args.answer)
    acc = corr.macro()
    E, T = en.mean(0), lat.mean(0)
    res = {"model": args.model, "answer": args.answer, "levels": L, "macro_acc": acc.tolist(),
           "energy_j": E.tolist(), "latency_ms": T.tolist(),
           "p90_latency_ms": np.percentile(lat, 90, axis=0).tolist()}

    print(f"== {args.model}  (answer length: {args.answer})")
    print("level   acc   J/ans  mean ms  p90 ms  MB")
    for j, k in enumerate(L):
        print(f"  L{k}  {acc[j]:5.1f}  {E[j]:5.2f}  {T[j]:7.0f}  {res['p90_latency_ms'][j]:6.0f}  "
              f"{prof.cost[k].weight_bytes/1e6:5.0f}")
    par, hull = pareto(E, acc), upper_hull(E, acc)
    res["pareto"] = [L[i] for i in par]
    res["hull_energy"] = [L[i] for i in hull]
    res["hull_latency"] = [L[i] for i in upper_hull(T, acc)]
    print("Pareto (energy):", res["pareto"], " hull (energy):", res["hull_energy"],
          " hull (latency):", res["hull_latency"])
    gap = {L[j]: round(float(acc[j] - hull_value(E, acc, hull, E[j])), 2) for j in range(len(L))}
    res["below_hull_pts"] = gap
    print("accuracy minus hull mix at the same energy:", gap)

    # ---- oracle per-question router (upper bound; uses the answers, not deployable)
    C = corr.C
    any_ok = C.any(1)
    first = np.where(any_ok, C.argmax(1), 0)            # widths sorted by cost: cheapest correct
    o_en = en[np.arange(len(first)), first].mean()
    per = [(any_ok & (corr.source == s)).sum() / (corr.source == s).sum() for s in range(len(D.SOURCES))]
    res["oracle"] = {"macro_acc": 100 * float(np.mean(per)), "energy_j": float(o_en),
                     "share_any_correct": float(any_ok.mean())}
    print(f"oracle router: macro {res['oracle']['macro_acc']:.1f} at {o_en:.2f} J/ans "
          f"(vs L8 {acc[-1]:.1f} at {E[-1]:.2f} J)")
    # headroom decomposition: answered by L8 only / by L0 only (guessing noise between widths)
    j8 = len(L) - 1
    res["overlap"] = {"L0_and_L8": float((C[:, 0] & C[:, j8]).mean()), "L0_not_L8": float((C[:, 0] & ~C[:, j8]).mean()),
                      "L8_not_L0": float((~C[:, 0] & C[:, j8]).mean())}
    print("overlap L0/L8 (share of questions):", {k: round(v, 3) for k, v in res["overlap"].items()})
    # Is the oracle headroom skill or luck? Compare the observed "any width correct" with what independent
    # answers of the same per-width accuracy would give (per test set, so option counts are respected).
    # Observed << independent means widths fail on the same questions (routable structure is small);
    # observed ~ independent means the oracle mostly harvests lucky guesses.
    indep, obs = [], []
    for s in range(len(D.SOURCES)):
        m = corr.source == s
        p = C[m].mean(0)
        indep.append(1 - np.prod(1 - p))
        obs.append(any_ok[m].mean())
    res["oracle_vs_independent"] = {"observed_macro": 100 * float(np.mean(obs)),
                                    "independent_macro": 100 * float(np.mean(indep))}
    print(f"any-width-correct: observed {100*np.mean(obs):.1f} vs {100*np.mean(indep):.1f} if widths erred "
          "independently")

    # ---- task-aware static choice: per test set, the width with best accuracy per joule on the hull,
    # chosen on the calibration half, scored on the evaluation half (no test leakage).
    cal = D.split_mask(len(corr.source), args.seed)
    A_cal, A_ev = corr.per_source(cal), corr.per_source(~cal)
    rows = []
    for s, name in enumerate(D.SHORT.values()):
        h = upper_hull(E, A_cal[s])
        rows.append({"set": name, "hull": [L[i] for i in h], "acc_eval_by_level": np.round(A_ev[s], 1).tolist()})
        print(f"  {name:8s} hull {[L[i] for i in h]}  eval acc by width {np.round(A_ev[s], 1).tolist()}")
    res["per_source"] = rows
    os.makedirs(os.path.join(D.HERE, "out"), exist_ok=True)
    out = os.path.join(D.HERE, "out", f"offline.{args.model.split('sft.')[-1]}.{args.answer}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
