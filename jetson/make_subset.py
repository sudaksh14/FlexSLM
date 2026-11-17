#!/usr/bin/env python3
"""Build the on-device question subset (run once, anywhere; the result is committed).

    python3 jetson/make_subset.py data/medmix_sft/test.jsonl jetson/questions/subset.jsonl --per-source 30

Stratified: --per-source questions of each of the seven test sets, seed 0. Each row keeps the
stored prompt, the gold answer (and MedCalc tolerances) and `test_index`, its position in the full
test file, which joins on-device answers with the cluster's per-question outputs.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mednest_jetson.workload import make_subset  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test_jsonl")
    p.add_argument("out_jsonl")
    p.add_argument("--per-source", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out_jsonl)), exist_ok=True)
    rows = make_subset(a.test_jsonl, a.out_jsonl, a.per_source, a.seed)
    print(f"{len(rows)} questions -> {a.out_jsonl}")
