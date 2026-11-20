"""Inputs of the width controller: per-question correctness at every width, and a Jetson cost profile.

Nothing here touches the training / eval / Jetson pipeline; it only reads copies of their outputs
(see README.md, "Inputs"):

- `data/samples/gen.<model>-level<k>.t32.samples.jsonl`: eval/eval_gen.py output, one line per test
  question, in data/medmix_sft/test.jsonl order (checked: sources match line by line).
- `data/jetson/widths_15W.jsonl`: jetson/bench_widths.py output (shape-only run, random weights,
  15 W, 210 questions, forced 32 new tokens). Latency/energy do not depend on the weight values.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")

SHORT = {"openlifescienceai/medqa": "MedQA", "openlifescienceai/medmcqa": "MCQA", "dvilares/head_qa": "HeadQA",
         "ncbi/MedCalc-Bench": "Calc", "JesseLiu/medbulltes5op": "Bullets", "TIGER-Lab/MMLU-Pro": "Pro-H",
         "TsinghuaC3I/MedXpertQA": "XpertQA"}
SOURCES = list(SHORT)

M3 = "hf_export-flexllama__sft.paper.qwen3_0p6b_nested"
# A2 (the paper's main 0.6B model: no Stage 1, 3 widths 188M/364M/596M). Its levels 0/1/2 have the shapes of grid
# levels 0/4/8, so its samples are copied under grid level numbers (level1 -> level4, level2 -> level8) and the
# 9-level Jetson profile applies unchanged. See README.md, Inputs.
A2 = "hf_export-flexllama__sft.paper.qwen3_0p6b_nested.no_stage1"
A2G = A2 + "_grid048"


def short_model(model: str) -> str:
    return {M3: "M3", A2G: "A2"}.get(model, model.rsplit(".", 1)[-1])


# ----------------------------------------------------------------------------- accuracy side
@dataclass
class Correctness:
    """C[q, j] = question q answered correctly at width levels[j]."""
    levels: List[int]
    C: np.ndarray                 # bool [Q, L]
    source: np.ndarray            # int  [Q] index into SOURCES
    prompt_chars: np.ndarray      # int  [Q]
    answer_chars: np.ndarray      # int  [Q, L] length of the generated text (for answer-length estimates)

    def macro(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Paper's 'Avg': mean over the 7 test sets of per-set accuracy, per width (in %)."""
        m = np.ones(len(self.source), bool) if mask is None else mask
        per = [self.C[m & (self.source == s)].mean(0) for s in range(len(SOURCES))]
        return 100.0 * np.mean(per, axis=0)

    def per_source(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """[S, L] accuracy in % per test set and width."""
        m = np.ones(len(self.source), bool) if mask is None else mask
        return 100.0 * np.stack([self.C[m & (self.source == s)].mean(0) for s in range(len(SOURCES))])


def load_correctness(model: str = M3, levels: Optional[List[int]] = None) -> Correctness:
    files = {}
    for f in glob.glob(os.path.join(DATA, "samples", f"gen.{model}-level*.t32.samples.jsonl")):
        k = int(re.search(r"-level(\d+)\.t32", f).group(1))
        files[k] = f
    if not files:
        raise FileNotFoundError(f"no samples for {model} in {DATA}/samples (see README.md, Inputs)")
    levels = sorted(files) if levels is None else levels
    test = [json.loads(l) for l in open(os.path.join(REPO, "data", "medmix_sft", "test.jsonl"), encoding="utf8")]
    cols, ans = [], []
    for k in levels:
        rows = [json.loads(l) for l in open(files[k], encoding="utf8")]
        if len(rows) != len(test) or any(r["source"] != t["source"] for r, t in zip(rows, test)):
            raise ValueError(f"{files[k]} is not in test.jsonl order")
        cols.append([bool(r["correct"]) for r in rows])
        ans.append([len(r.get("gen") or "") for r in rows])
    return Correctness(levels=levels, C=np.array(cols).T, source=np.array([SOURCES.index(t["source"]) for t in test]),
                       prompt_chars=np.array([len(t["prompt"]) for t in test]), answer_chars=np.array(ans).T)


# ----------------------------------------------------------------------------- cost side
@dataclass
class LevelCost:
    level: int
    params: int
    weight_bytes: int             # bytes of this width alone (a dense / compiled copy)
    ttft_a: float                 # TTFT(ms) = a + b * prompt_tokens  (least squares over the run)
    ttft_b: float
    tpot_ms: float                # mean decode ms per new token
    net_power_w: float            # net (above idle) board power while serving this width
    kv_cache_bytes: int


@dataclass
class Profile:
    mode: str
    levels: List[int]
    cost: Dict[int, LevelCost]
    resident_bytes: int           # all widths as views of one weight set
    ratio_samples: np.ndarray     # observed / predicted total latency (noise model)
    chars_per_token: float        # prompt chars -> tokens (fit on the joined questions)
    switch_view_ms: float = 0.037
    switch_load_ms: float = 900.0  # slice-load / reload of one dense width (0.75-1.1 s measured)
    extra: dict = field(default_factory=dict)

    def latency_ms(self, level: int, prompt_tokens, new_tokens) -> np.ndarray:
        c = self.cost[level]
        return c.ttft_a + c.ttft_b * np.asarray(prompt_tokens) + c.tpot_ms * np.asarray(new_tokens)

    def energy_j(self, level: int, latency_ms) -> np.ndarray:
        return self.cost[level].net_power_w * np.asarray(latency_ms) / 1000.0


def load_profile(path: Optional[str] = None, mode: str = "15W") -> Profile:
    path = path or os.path.join(DATA, "jetson", "widths_15W.jsonl")
    ans, summ = {}, {}
    for l in open(path, encoding="utf8"):
        r = json.loads(l)
        if r.get("target") != "mednest":
            continue
        if r.get("kind") == "answer":
            ans.setdefault(r["level"], []).append(r)
        elif r.get("kind") == "summary":
            summ[r["level"]] = r
    test_path = os.path.join(REPO, "data", "medmix_sft", "test.jsonl")
    # the test set is only needed for the chars -> tokens fit (simulation); the board tokenises itself
    test = [json.loads(l) for l in open(test_path, encoding="utf8")] if os.path.exists(test_path) else None
    cost, ratios, chars, toks = {}, [], [], []
    for k in sorted(summ):
        rows, s = ans[k], summ[k]
        p = np.array([r["prompt_tokens"] for r in rows], float)
        t = np.array([r["ttft_ms"] for r in rows], float)
        b, a = np.polyfit(p, t, 1)
        tpot = float(np.mean([r["ms_per_decode_token"] for r in rows]))
        busy_s = sum(r["total_ms"] for r in rows) / 1000.0
        cost[k] = LevelCost(level=k, params=s["params"], weight_bytes=s["weight_bytes_width"], ttft_a=float(a),
                            ttft_b=float(b), tpot_ms=tpot, net_power_w=s["energy_net_j"] / busy_s,
                            kv_cache_bytes=s.get("kv_cache_bytes", 0))
        pred = a + b * p + tpot * np.array([r["n_new"] for r in rows])
        ratios += list(np.array([r["total_ms"] for r in rows]) / pred)
        if test is not None:
            for r in rows:
                if r.get("test_index") is not None:
                    chars.append(len(test[r["test_index"]]["prompt"]))
                    toks.append(r["prompt_tokens"])
    if not summ:
        raise ValueError(f"{path}: no mednest summary records")
    any_s = next(iter(summ.values()))
    return Profile(mode=mode, levels=sorted(cost), cost=cost, resident_bytes=any_s["weight_bytes_full"],
                   ratio_samples=np.array(ratios),
                   chars_per_token=float(np.sum(chars) / np.sum(toks)) if toks else float("nan"))


def answer_tokens(corr: Correctness, how: str = "eos") -> np.ndarray:
    """New tokens per (question, width). 'forced32' = the Jetson bench setting; 'eos' = generation stops
    after the answer: ~3.2 chars/token for '\\boxed{B}'-style text, +1 for the end token, at most 32."""
    if how == "forced32":
        return np.full(corr.answer_chars.shape, 32)
    return np.minimum(32, np.ceil(corr.answer_chars / 3.2).astype(int) + 1)


def prompt_tokens(corr: Correctness, prof: Profile) -> np.ndarray:
    return np.maximum(1, np.round(corr.prompt_chars / prof.chars_per_token)).astype(int)


def split_mask(n: int, seed: int = 0) -> np.ndarray:
    """True = calibration half (controller statistics), False = evaluation half."""
    return np.random.default_rng(seed).random(n) < 0.5


if __name__ == "__main__":
    corr = load_correctness()
    prof = load_profile()
    print("levels", corr.levels, "questions", len(corr.source))
    print("macro acc", np.round(corr.macro(), 1))
    print("chars/token", round(prof.chars_per_token, 2), "noise ratio p10/p50/p90",
          np.round(np.percentile(prof.ratio_samples, [10, 50, 90]), 3))
    for k, c in prof.cost.items():
        print(f"L{k}: ttft {c.ttft_a:.0f}+{c.ttft_b:.3f}*p ms, {c.tpot_ms:.1f} ms/tok, net {c.net_power_w:.2f} W, "
              f"{c.weight_bytes/1e6:.0f} MB")
