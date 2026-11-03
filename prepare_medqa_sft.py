import argparse
import collections
import json
import os
import random
import re

import pandas as pd
from huggingface_hub import hf_hub_download

MEDQA_REPO = "GBaker/MedQA-USMLE-4-options-hf"
MEDMCQA_REPO = "openlifescienceai/medmcqa"
MMLU_REPO = "cais/mmlu"
MMLU_MED = ["anatomy", "clinical_knowledge", "college_medicine",
            "medical_genetics", "professional_medicine", "virology"]
NGRAM = 13


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def ngrams(text):
    words = norm(text).split()
    return {" ".join(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def option_key(question, options):
    return norm(question) + " || " + " | ".join(sorted(norm(o) for o in options))


def fetch(repo, filename, cache_dir):
    return hf_hub_download(repo, filename, repo_type="dataset", cache_dir=cache_dir)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_items(cache_dir):
    """Returns (train, evals): lists of dicts with question/options/target/source."""
    def medqa(split):
        rows = read_jsonl(fetch(MEDQA_REPO, f"{split}.json", cache_dir))
        return [dict(question=r["sent1"], options=[r[f"ending{i}"] for i in range(4)],
                     target=int(r["label"]), source=f"medqa_{split}") for r in rows]

    def medmcqa(split):
        df = pd.read_parquet(fetch(MEDMCQA_REPO, f"data/{split}-00000-of-00001.parquet", cache_dir))
        return [dict(question=r.question, options=[r.opa, r.opb, r.opc, r.opd],
                     target=int(r.cop), source=f"medmcqa_{split}") for r in df.itertuples()]

    train = medqa("train") + medmcqa("train")
    evals = medqa("test") + medmcqa("validation")
    for subject in MMLU_MED:
        df = pd.read_parquet(fetch(MMLU_REPO, f"{subject}/test-00000-of-00001.parquet", cache_dir))
        evals += [dict(question=r.question, options=list(r.choices), target=int(r.answer),
                       source=f"mmlu_{subject}") for r in df.itertuples()]
    return train, evals


def find_contaminated(train, evals):
    """Returns {train index: reason} for train items that near-duplicate an eval item."""
    eval_keys = {option_key(e["question"], e["options"]) for e in evals}
    eval_grams = [ngrams(e["question"]) for e in evals]
    gram_index = collections.defaultdict(list)
    for j, grams in enumerate(eval_grams):
        for g in grams:
            gram_index[g].append(j)

    dropped = {}
    for i, t in enumerate(train):
        if option_key(t["question"], t["options"]) in eval_keys:
            dropped[i] = "exact question+options"
            continue
        shared = collections.Counter()
        for g in ngrams(t["question"]):
            for j in gram_index.get(g, ()):
                shared[j] += 1
        for j, n in shared.items():
            if n / len(eval_grams[j]) >= 0.5:
                dropped[i] = f"13-gram overlap with {evals[j]['source']}"
                break
    return dropped


def to_example(item):
    return {"prompt": f"Question: {item['question']}\nAnswer:",
            "answer": " " + str(item["options"][item["target"]]),
            "source": item["source"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/medqa_sft")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=None, help="HF hub cache (default: HF_HOME)")
    args = parser.parse_args()

    train, evals = load_items(args.cache_dir)
    dropped = find_contaminated(train, evals)
    kept = [to_example(t) for i, t in enumerate(train) if i not in dropped]

    random.Random(args.seed).shuffle(kept)
    n_val = int(len(kept) * args.val_frac)
    val, train_out = kept[:n_val], kept[n_val:]

    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("train", train_out), ("val", val)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    report = {
        "train_items_in": len(train),
        "eval_items_checked": len(evals),
        "dropped": len(dropped),
        "dropped_by_reason": collections.Counter(dropped.values()),
        "dropped_examples": [train[i]["question"][:200] for i in list(dropped)[:10]],
        "train_out": len(train_out),
        "val_out": len(val),
        "train_out_by_source": collections.Counter(r["source"] for r in train_out),
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_examples"}, indent=2))


if __name__ == "__main__":
    main()
