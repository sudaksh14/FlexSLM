"""Two separate models vs nested models: what a device owner would ship instead of one nested model.

    python controller/two_model.py            # writes out/two_model.txt

Everything comes from simulate.py's world (same questions, calibration / evaluation split, 15 W profile, noise),
extended to arms of two backbones:
  0.6B nested (A2):        188M / 364M / 596M            profile data/jetson/widths_15W.jsonl
  1.7B nested (run 4):     8 of its 10 widths            profile data/jetson/widths_15W_1p7b.jsonl
  conventional FT:         0.6B (596M) and the 1.7B teacher of run 4 (1.72B), same shapes as the full widths
Run 4's widths use the closest profiled shape (parameters within 3%); 0.24B and 0.40B have none and are left out.
Deployments:
  N06     nested 0.6B, all 3 widths                      (1.19 GB of weights)
  N17     nested 1.7B, 8 widths                          (3.44 GB)
  PAIR    the two MedNest full widths, 596M + 1.72B      (4.63 GB, both resident)
  PAIRc   conventional FT 596M + 1.72B                   (4.63 GB, both resident)
  BOTH    all widths of both nested models               (4.63 GB)
  *-load  one model resident at a time: a switch costs a load (0.6B 650 ms = middle of the measured warm 0.47-0.82 s;
          1.7B scaled by weight bytes, NOT measured; the energy of the load is not counted)
          Limit: the rule is myopic and the assumed 1.7B load (1.9 s) exceeds every deadline, so once it drops to 596M it never returns.
          Read *-load only as 'no per-request switching', not as a tuned one-model-at-a-time policy.
Simulated, not measured on a device: co-residency is ASSUMED to fit (on the board a 1.0 GB model did not load next to the
3.44 GB one, in one process that had released nothing; see the log).
Policies: deadline rules choose the most accurate arm (calibration half) whose predicted latency meets the deadline (the fastest
if none does), so unlike DeadlinePolicy they also work when arms of two models interleave in cost and accuracy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import data_io as D
import simulate as sim
from offline import costs, upper_hull, hull_value
from policies import EnergyQueuePolicy, Policy, Request, _Predictor

A2G = D.A2G
RUN4 = "student1p7b_grid"                # samples of run 4 stored under profile level numbers
CONV06 = "hf_export-flexllama__sft.paper.qwen3_0p6b_conventional"
CONV17 = "conv1p7b_teacher"
RUN4_PARAMS = {0: 497, 1: 619, 2: 745, 3: 900, 4: 1051, 6: 1332, 7: 1527, 8: 1721}   # profile level -> M params
LOAD_MS_06, GB06, GB17 = 650.0, 1.19, 3.44
LOAD = {"0.6": LOAD_MS_06, "1.7": LOAD_MS_06 * GB17 / GB06}


def build():
    """Arms as one combined Correctness + Profile. Arm ids are consecutive integers; NAMES[arm] = (family, label)."""
    p06 = D.load_profile()
    p17 = D.load_profile(os.path.join(D.DATA, "jetson", "widths_15W_1p7b.jsonl"))
    c06, c17 = D.load_correctness(A2G), D.load_correctness(RUN4)
    k06, k17 = D.load_correctness(CONV06, [2]), D.load_correctness(CONV17, [9])
    arms, corrs, cost, names = [], [], {}, {}
    parts = [(c06, p06, "0.6", [(j, l, f"{ {0: 188, 4: 364, 8: 596}[l] }M") for j, l in enumerate(c06.levels)]),
             (c17, p17, "1.7", [(j, l, f"{RUN4_PARAMS[l]}M") for j, l in enumerate(c17.levels)]),
             (k06, p06, "0.6c", [(0, 8, "596Mc")]), (k17, p17, "1.7c", [(0, 8, "1721Mc")])]
    C, ans = [], []
    for corr, prof, fam, items in parts:
        for j, l, label in items:
            a = len(arms)
            arms.append(a)
            cost[a] = prof.cost[l]
            names[a] = (fam, label)
            C.append(corr.C[:, j])
            ans.append(corr.answer_chars[:, j])
    corr = D.Correctness(levels=arms, C=np.array(C).T, source=c06.source, prompt_chars=c06.prompt_chars,
                         answer_chars=np.array(ans).T)
    prof = D.Profile(mode="15W", levels=arms, cost=cost, resident_bytes=0,
                     ratio_samples=np.concatenate([p06.ratio_samples, p17.ratio_samples]),
                     chars_per_token=p06.chars_per_token)
    return corr, prof, names


class W:
    """Subset of sim.World: a chosen set of arms."""
    def __init__(self, corr, prof, arms, seed=0):
        self.corr = D.Correctness(levels=list(arms), C=corr.C[:, arms], source=corr.source,
                                  prompt_chars=corr.prompt_chars, answer_chars=corr.answer_chars[:, arms])
        self.prof, self.L = prof, list(arms)
        self.cal = D.split_mask(len(corr.source), seed)
        self.ptok = D.prompt_tokens(self.corr, prof)
        self.ntok = D.answer_tokens(self.corr, "eos")
        self.lat, self.en = costs(self.corr, prof, "eos")
        self.acc_cal = self.corr.macro(self.cal)
        self.acc_task_cal = sim.task_table(self.corr.per_source(self.cal), self.acc_cal, self.corr, self.cal, "shrunk")
        self.n_pred = float(np.percentile(self.ntok[self.cal], 90))
        self.E_cal = self.en[self.cal].mean(0)
        self.eval_idx = np.flatnonzero(~self.cal)

    def stream(self, seed=0):
        return np.random.default_rng(seed).permutation(self.eval_idx)


class Deadline(Policy):
    """Most accurate arm whose predicted latency (with feedback rho and, if given, the load cost of switching) meets the
    deadline, else the arm with the smallest predicted latency."""
    def __init__(self, w: W, feedback=True, load=None, names=None):
        self.w, self.load, self.names = w, load, names
        self.pred = _Predictor(w.prof, w.L, w.n_pred, feedback, 0.2)
        self.acc = dict(zip(w.L, w.acc_cal))
        self.cur = None
        self.name = "deadline" + ("+fb" if feedback else "")

    def _pred(self, l, p):
        t = self.pred.latency(l, p)
        if self.load and self.cur is not None and self.names[l][0] != self.names[self.cur][0]:
            t += self.load[self.names[l][0][:3]]
        return t

    def choose(self, req: Request) -> int:
        t = {l: self._pred(l, req.prompt_tokens) for l in self.w.L}
        ok = [l for l in self.w.L if req.deadline_ms is None or t[l] <= req.deadline_ms]
        self.cur = max(ok, key=lambda l: (self.acc[l], -t[l])) if ok else min(self.w.L, key=lambda l: t[l])
        return self.cur

    def observe(self, level, latency_ms, energy_j):
        # the predictor's raw prediction is for the current arm; load time is not part of it, so it is subtracted by run()
        self.pred.update(level, latency_ms)


def run(w: W, pol: Policy, qs, names, deadlines=None, slowdown=None, tags=None, load=None, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.choice(w.prof.ratio_samples, size=len(qs))
    idx = {l: j for j, l in enumerate(w.L)}
    recs, prev = [], None
    for i, q in enumerate(qs):
        dl = deadlines[i] if deadlines is not None else None
        lvl = pol.choose(Request(prompt_tokens=int(w.ptok[q]), deadline_ms=dl, task=int(w.corr.source[q])))
        j, s = idx[lvl], (slowdown[i] if slowdown is not None else 1.0)
        serve = float(w.lat[q, j] * noise[i] * s)
        sw = 0.0
        if prev is not None and prev != lvl:
            same = names[prev][0] == names[lvl][0]
            sw = load[names[lvl][0][:3]] if (load and not same) else w.prof.switch_view_ms
        en = float(w.prof.energy_j(lvl, serve))
        pol.observe(lvl, serve, en)
        recs.append((int(w.corr.source[q]), bool(w.corr.C[q, j]), serve + sw, en, lvl, dl, tags[i] if tags else None))
        prev = lvl
    out = sim.summarize(pol.name, recs)
    out["_correct"], out["_source"] = np.array([r[1] for r in recs]), np.array([r[0] for r in recs])
    return out


def main():
    corr, prof, names = build()
    by = {v[1]: k for k, v in names.items()}
    fam = lambda f: [a for a, v in names.items() if v[0] == f]
    deps = {"N06": fam("0.6"), "N17": fam("1.7"), "PAIR": [by["596M"], by["1721M"]],
            "PAIRc": [by["596Mc"], by["1721Mc"]], "BOTH": fam("0.6") + fam("1.7")}
    gb = {"N06": GB06, "N17": GB17, "PAIR": GB06 + GB17, "PAIRc": GB06 + GB17, "BOTH": GB06 + GB17}
    # arm table on all questions
    allw = W(corr, prof, sorted(names))
    E, acc = allw.en.mean(0), corr.macro()
    print("arm (all 12,797 q)      acc     J/answer  ms")
    for j, a in enumerate(allw.L):
        print(f"  {names[a][0]:5s}{names[a][1]:7s} {acc[j]:6.1f} {E[j]:8.2f} {allw.lat.mean(0)[j]:7.0f}")
    print("\nenergy hulls (all questions):")
    for d, arms in deps.items():
        js = [allw.L.index(a) for a in arms]
        h = [js[i] for i in upper_hull(E[js], acc[js])]
        print(f"  {d:6s} {gb[d]:.2f} GB  " + "  ".join(f"{names[allw.L[i]][1]}({E[i]:.2f} J, {acc[i]:.1f})" for i in h))

    qs = None
    ws = {d: W(corr, prof, arms) for d, arms in deps.items()}
    qs = ws["PAIR"].stream()
    print(f"\n== budget ({len(qs)} requests, evaluation half, task-aware queue, V 10): acc (J used); '-' = cheapest arm exceeds the budget")
    print("  budget  " + "".join(f"{d:>18s}" for d in deps))
    B_list = [1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4]
    res = {}
    for B in B_list:
        row = f"  {B:4.1f} J  "
        for d, w in ws.items():
            if w.E_cal.min() > B:
                row += f"{'-':>18s}"
                continue
            p = EnergyQueuePolicy(w.prof, w.L, w.acc_task_cal, budget_j=B, V=10.0, answer_tokens=w.n_pred, feedback=False, task_aware=True)
            r = run(w, p, qs, names)
            res[d, B] = r
            over = "!" if r["energy_j"] > B * 1.01 else ""
            row += f"{r['macro_acc']:10.1f} ({r['energy_j']:.2f}{over})"
        print(row)
    print("\n  paired differences (95% bootstrap CI over the same requests)")
    for B in B_list:
        for a, b in [("PAIR", "N17"), ("PAIR", "PAIRc"), ("BOTH", "PAIR"), ("PAIR", "N06")]:
            if (a, B) in res and (b, B) in res:
                d = sim.paired_ci(res[a, B], res[b, B], n_boot=1000)
                print(f"  {B:.1f} J  {a} - {b}: {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")

    print("\n== deadlines: 30% urgent, routine 1206 ms, middle third x1.3 slower, deadline rule with feedback; acc / miss% (urgent miss%) / J")
    n = len(qs)
    u = np.random.default_rng(1).random(n) < 0.3
    slow = [1.3 if n // 3 <= i < 2 * (n // 3) else 1.0 for i in range(n)]
    tags = [("urgent" if x else "routine") + ("-slow" if s > 1 else "") for x, s in zip(u, slow)]
    cfgs = [(d, None) for d in deps] + [("PAIR", LOAD), ("PAIRc", LOAD)]
    print("  urgent deadline  " + "".join(f"{d + ('-load' if l else ''):>22s}" for d, l in cfgs))
    dres = {}
    for U in [300, 400, 500, 600]:
        dls = [float(U) if x else 1206.0 for x in u]
        row = f"  {U:4d} ms          "
        for d, l in cfgs:
            w = ws[d]
            r = run(w, Deadline(w, True, l, names), qs, names, dls, slow, tags, l)
            dres[d, bool(l), U] = r
            um = np.mean([r["miss_by_tag"][t] for t in ("urgent", "urgent-slow")])
            row += f"{r['macro_acc']:8.1f}/{100*r['miss_rate']:4.1f}({100*um:4.1f})/{r['energy_j']:.2f}"
        print(row)
    print("\n  switches and arm use at urgent 500 ms:")
    for d, l in cfgs:
        r = dres[d, bool(l), 500]
        print(f"  {d + ('-load' if l else ''):12s} switches {r['switches']:5d}  " + "  ".join(f"{names[a][1]}:{c}" for a, c in r["levels"].items()))
    for a, b, l in [("PAIR", "N17", False), ("PAIR", "N17", True), ("BOTH", "PAIR", False)]:
        for U in (400, 500):
            x, y = dres[a, l, U], dres[b, False, U]
            d = sim.paired_ci(x, y, n_boot=1000)
            print(f"  urgent {U} ms  {a}{'-load' if l else ''} - {b}: acc {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]  miss {100*(x['miss_rate']-y['miss_rate']):+.1f}")


if __name__ == "__main__":
    main()
