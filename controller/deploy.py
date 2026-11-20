"""Tier 1: deploy-time selection. Given a device budget, decide what goes on the device.

    python controller/deploy.py --backend views    --weight-mb 7000
    python controller/deploy.py --backend compiled --weight-mb 1600 --energy-j 1.8
    python controller/deploy.py --backend compiled --weight-mb 700  --p90-ms 400

backend=views:    the elastic model (all widths are views of one weight set, 0.037 ms switch); the runtime
                  controller (policies.py) chooses per request among the hull widths.
backend=compiled: fixed-shape engines (TensorRT / llama.cpp GGUF) share no weights and switch by reloading
                  (~1 s), so only 1-2 anchor widths are materialised; each is a dense Qwen3 of that width
                  (hf_export/<cfg>/level<k>, or the legacy delta file networks/level_delta_utils.py). Two
                  resident anchors are switched for free by the same energy queue.
The profile is the Jetson Orin Nano 15 W one; accuracy is the calibration-half macro average.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import data_io as D
from offline import costs
from policies import deploy_select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["views", "compiled"], required=True)
    ap.add_argument("--weight-mb", type=float, required=True, help="memory available for weights")
    ap.add_argument("--p90-ms", type=float, default=None)
    ap.add_argument("--energy-j", type=float, default=None, help="long-run budget, J per answer")
    ap.add_argument("--answer", default="eos", choices=["eos", "forced32"])
    ap.add_argument("--max-anchors", type=int, default=2)
    args = ap.parse_args()
    corr, prof = D.load_correctness(), D.load_profile()
    cal = D.split_mask(len(corr.source))
    lat, en = costs(corr, prof, args.answer)
    m = deploy_select(prof, list(corr.macro(cal)), corr.levels, backend=args.backend,
                      weight_budget_bytes=args.weight_mb * 1e6, p90_deadline_ms=args.p90_ms,
                      energy_budget_j=args.energy_j, mean_energy=list(en[cal].mean(0)),
                      p90_latency=list(np.percentile(lat[cal], 90, axis=0)), max_anchors=args.max_anchors)
    m["inputs"] = vars(args)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
