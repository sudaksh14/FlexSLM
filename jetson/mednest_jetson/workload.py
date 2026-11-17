"""Workload for the on-device runs: a fixed, stratified subset of the seven test sets, prompt
rendering and scoring exactly as in the cluster evaluation (eval/eval_gen.py), and a timed greedy
decoder (batch 1, KV cache) that reports time to first token and per-token decode time.

The decoder is HF greedy decoding written out (prefill, then one token at a time with the KV
cache), so TTFT and decode time can be separated; its answers equal model.generate(do_sample=False)
up to kernel-level numerics (test/test_jetson_elastic.py checks this on CPU).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from eval.eval_gen import MEDCALC, forced_answer, is_correct, last_boxed, render_prompt  # noqa: E402

SOURCES = ("openlifescienceai/medqa", "openlifescienceai/medmcqa", "dvilares/head_qa", "ncbi/MedCalc-Bench",
           "JesseLiu/medbulltes5op", "TIGER-Lab/MMLU-Pro", "TsinghuaC3I/MedXpertQA")
SHORT = {"openlifescienceai/medqa": "MedQA", "openlifescienceai/medmcqa": "MCQA", "dvilares/head_qa": "HeadQA",
         "ncbi/MedCalc-Bench": "Calc", "JesseLiu/medbulltes5op": "Bullets", "TIGER-Lab/MMLU-Pro": "Pro-H",
         "TsinghuaC3I/MedXpertQA": "XpertQA"}


def make_subset(test_jsonl: str, out_jsonl: str, per_source: int = 20, seed: int = 0) -> List[dict]:
    """Stratified subset (per_source questions of each test set, seeded). Each row keeps its index
    in the full test file (`test_index`), so on-device answers can be joined with the cluster's
    per-question outputs (results/gen.*.samples.jsonl are in test-file order)."""
    rows = [json.loads(l) for l in open(test_jsonl)]
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["source"]].append(i)
    rng = random.Random(seed)
    keep = []
    for src in SOURCES:
        idx = sorted(rng.sample(by[src], min(per_source, len(by[src]))))
        keep += idx
    with open(out_jsonl, "w") as f:
        for i in keep:
            f.write(json.dumps(dict(rows[i], test_index=i)) + "\n")
    return [dict(rows[i], test_index=i) for i in keep]


def load_subset(path: str, limit: Optional[int] = None) -> List[dict]:
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def prompt_for(row: dict, fmt: str, tokenizer=None) -> str:
    return render_prompt(row["prompt"], fmt, tokenizer)


def score(row: dict, gen: str, fmt: str) -> Dict[str, object]:
    scored = (forced_answer(gen, row) or "") if fmt == "apollo_answer" else gen
    return {"correct": bool(is_correct(row, scored)), "boxed": last_boxed(scored)}


def eos_ids(model, tokenizer) -> List[int]:
    ids = model.generation_config.eos_token_id
    ids = list(ids) if isinstance(ids, (list, tuple)) else ([ids] if ids is not None else [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def timed_greedy(model, input_ids: torch.Tensor, max_new_tokens: int, eos: List[int]) -> dict:
    """Greedy decode of one prompt (batch 1). Returns the new token ids and timings in ms:
    ttft_ms (prefill + first token), decode_ms (all later tokens), total_ms, per-token mean."""
    from transformers import DynamicCache
    _sync()
    t0 = time.perf_counter()
    cache = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1].argmax(-1)
    _sync()
    t1 = time.perf_counter()
    new = [nxt.item()]
    cache = out.past_key_values
    while len(new) < max_new_tokens and new[-1] not in eos:
        out = model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        new.append(nxt.item())     # .item() synchronises: every token is really done here
    _sync()
    t2 = time.perf_counter()
    n_dec = len(new) - 1
    return {"new_ids": new, "n_new": len(new), "prompt_tokens": int(input_ids.shape[1]),
            "ttft_ms": (t1 - t0) * 1000.0, "decode_ms": (t2 - t1) * 1000.0, "total_ms": (t2 - t0) * 1000.0,
            "ms_per_decode_token": ((t2 - t1) * 1000.0 / n_dec) if n_dec else None,
            "stopped_on_eos": new[-1] in eos}


def encode(tokenizer, text: str, fmt: str, device) -> torch.Tensor:
    # Same tokenisation as eval_gen.generate: stored prompts and chat templates carry their own
    # special tokens; the plain Apollo format gets the tokenizer's BOS.
    return tokenizer(text, return_tensors="pt", add_special_tokens=fmt.startswith("apollo")).input_ids.to(device)


def answer(model, tokenizer, row: dict, fmt: str, max_new_tokens: int, device, decoder=None) -> dict:
    """Prompt -> timed greedy answer -> scored record. decoder: a decode.GraphDecoder for `model`
    (static KV cache + CUDA-graph steps, the default on the board); None = eager HF decoding."""
    ids = encode(tokenizer, prompt_for(row, fmt, tokenizer), fmt, device)
    eos = eos_ids(model, tokenizer)
    r = decoder.generate(ids, max_new_tokens, eos) if decoder is not None else \
        dict(timed_greedy(model, ids, max_new_tokens, eos), decoder="eager")
    gen = tokenizer.decode(r.pop("new_ids"), skip_special_tokens=True)
    return {"test_index": row.get("test_index"), "source": row["source"], "gt": row["gt"], "gen": gen,
            **score(row, gen, fmt), **r}


def accuracy_by_source(records: List[dict]) -> Dict[str, float]:
    by = defaultdict(list)
    for r in records:
        by[SHORT.get(r["source"], r["source"])].append(r["correct"])
    accs = {k: 100.0 * sum(v) / len(v) for k, v in by.items()}
    if accs:
        accs["Avg"] = sum(accs.values()) / len(accs)
    return accs
