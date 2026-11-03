"""
FaithMed-style generated-answer evaluation of a Hugging Face model (a raw HF
model, or one FlexLLaMA level exported with export_flex_hf.py) on the medmix test
sets written by prepare_medmix_sft.py. Greedy decoding, then FaithMed's scoring
rule (rewards/medmix_reward.py), re-implemented here:
  - multiple choice: the last \\boxed{...} must contain exactly one option letter,
    which must equal the gold letter;
  - MedCalc-Bench: decimals must fall inside the gold [lower, upper] limits,
    integers must match after rounding, dates and (weeks, days) must match exactly.
Prompts are rendered with thinking disabled (no search, no reasoning trace).

    python eval_gen.py hf_export/<config>/level2 --out results/gen.<name>.level2.json
"""
import argparse
import collections
import json
import math
import re

import torch

BOXED = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
MEDCALC = "ncbi/MedCalc-Bench"


def last_boxed(text: str):
    found = BOXED.findall(text)
    return found[-1].strip() if found else None


def score_choice(gen: str, gt: str) -> bool:
    box = last_boxed(gen)
    if box is None:
        return False
    letters = set(re.findall(r"\b([A-J])\b", box))
    return len(letters) == 1 and letters.pop() == gt.strip().upper()


def _number(s: str):
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return None


def score_medcalc(gen: str, row: dict) -> bool:
    box = last_boxed(gen)
    if box is None:
        return False
    kind = (row.get("output_type") or "").lower()
    if kind == "decimal":
        x, lo, hi = _number(box), _number(str(row["lower"])), _number(str(row["upper"]))
        return None not in (x, lo, hi) and lo <= x <= hi
    if kind == "integer":
        x, gt = _number(box), _number(row["gt"])
        return None not in (x, gt) and round(x) == round(gt)
    norm = lambda s: re.sub(r"\s+", "", s).lstrip("0").replace("/0", "/")
    return norm(box) == norm(row["gt"])


def is_correct(row: dict, gen: str) -> bool:
    return score_medcalc(gen, row) if row["source"] == MEDCALC else score_choice(gen, row["gt"])


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size: int, max_new_tokens: int):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t
        print(f"  {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF model id or local directory")
    parser.add_argument("--data", default="data/medmix_sft/test.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default=None, help="comma-separated subset of test sources")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    rows = [json.loads(line) for line in open(args.data)]
    if args.sources:
        keep = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
    gens = generate(model, tokenizer, [r["prompt"] for r in rows], args.batch_size, args.max_new_tokens)

    per = collections.defaultdict(list)
    samples = []
    for r, g in zip(rows, gens):
        ok = is_correct(r, g)
        per[r["source"]].append(ok)
        samples.append({"source": r["source"], "gt": r["gt"], "boxed": last_boxed(g), "correct": ok, "gen": g})

    results = {}
    for src, oks in sorted(per.items()):
        acc = sum(oks) / len(oks)
        results[src] = {"acc": acc, "stderr": math.sqrt(acc * (1 - acc) / len(oks)), "n": len(oks),
                        "no_boxed": sum(s["boxed"] is None for s in samples if s["source"] == src)}
        print(f"{src:28s} acc {100 * acc:5.1f} ± {100 * results[src]['stderr']:.1f}  "
              f"(n {len(oks)}, no \\boxed {results[src]['no_boxed']})")
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "max_new_tokens": args.max_new_tokens, "results": results}, f, indent=2)
    with open(args.out.replace(".json", ".samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
