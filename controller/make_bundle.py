"""Build the request bundle the Jetson runner needs (run on a machine with the cluster samples copied).

    python controller/make_bundle.py [--n 600] [--seed 0] [--model <samples prefix>]
    -> controller/bundles/<name>.json   (tracked; small; synced with the code)

Contents:
- the controller's tables, from the CALIBRATION half only: macro accuracy per width, per-test-set accuracy
  per width (shrunk toward the global table, as in simulate.py), p90 answer tokens;
- the request stream: n // 7 questions per test set from the EVALUATION half, in a seeded order (prompt, source, gt,
  test_index), each with the cluster's answer text and correctness at every width. The runner uses
  those to score requests when the board runs random-weight exports (--score cluster) and to check
  agreement when it runs the trained export (--score device).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import data_io as D
from simulate import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=D.M3)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-table", default="shrunk", choices=["raw", "shrunk", "scaled"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--uniform", action="store_true", help="uniform over the evaluation half (default: stratified)")
    args = ap.parse_args()

    w = World("eos", args.seed, args.model, task_mode=args.task_table)
    test = [json.loads(l) for l in open(os.path.join(D.REPO, "data", "medmix_sft", "test.jsonl"), encoding="utf8")]
    gens = {}
    for k in w.L:
        f = os.path.join(D.DATA, "samples", f"gen.{args.model}-level{k}.t32.samples.jsonl")
        gens[k] = [json.loads(l).get("gen") or "" for l in open(f, encoding="utf8")]
    if args.uniform:
        qs = w.stream(args.n, args.seed)
    else:   # n // 7 per test set (macro accuracy weights the sets equally), then one seeded shuffle
        rng = np.random.default_rng(args.seed)
        per = args.n // len(D.SOURCES)
        qs = np.concatenate([rng.choice(w.eval_idx[w.corr.source[w.eval_idx] == s], per, replace=False)
                             for s in range(len(D.SOURCES))])
        qs = rng.permutation(qs)
    reqs = [{"test_index": int(q), "source": test[q]["source"], "gt": test[q]["gt"], "prompt": test[q]["prompt"],
             "correct": [bool(w.corr.C[q, j]) for j in range(len(w.L))], "gen": [gens[k][q] for k in w.L]}
            for q in qs]
    bundle = {"kind": "controller_bundle", "model": args.model, "seed": args.seed, "split": "data_io.split_mask",
              "task_table": args.task_table, "levels": w.L, "sources": D.SOURCES,
              "acc_cal": [float(x) for x in w.acc_cal], "acc_task_cal": w.acc_task_cal.round(4).tolist(),
              "n_pred_tokens": w.n_pred, "requests": reqs}
    os.makedirs(os.path.join(D.HERE, "bundles"), exist_ok=True)
    name = args.name or f"{args.model.split('sft.')[-1]}.n{args.n}.s{args.seed}"
    out = os.path.join(D.HERE, "bundles", name + ".json")
    json.dump(bundle, open(out, "w", encoding="utf8"))
    print(f"{len(reqs)} requests, levels {w.L}, sources {np.bincount([D.SOURCES.index(r['source']) for r in reqs])}"
          f" -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
