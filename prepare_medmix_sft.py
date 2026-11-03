"""
FaithMed's "medmix" data (github.com/cxcscmu/FaithMed, data_preparation/medmix_preparation.py)
rebuilt 1:1 for SFT of a Qwen3-backbone FlexLLaMA: same sources, splits, prompt
template, seed-42 shuffle, 16-gram test->train decontamination (MedCalc exempt) and
10k-per-source train cap, in that order. Prompts are rendered with the Qwen3 chat
template (thinking disabled); the SFT target is the boxed final answer.

Differences from FaithMed, all forced by what is available to us:
  - no reasoning trajectories (theirs were generated with DeepSeek-R1 / Claude and
    are not released), so the target is the answer only;
  - a small validation split (--val_size) is held out from the capped train set,
    because the FlexSLM trainer needs one for checkpointing;
  - HEAD-QA is read straight from its zip, since the HF loading script no longer runs.

    python prepare_medmix_sft.py              # writes data/medmix_sft/
"""
import argparse
import ast
import collections
import json
import os
import random
import zipfile

import datasets
from huggingface_hub import hf_hub_download

QA_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions.
QUESTION: {task_description}

Let's think step by step and include the final choice in the box \\boxed{{}}."""

CALC_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions.
QUESTION: {task_description}

When you are finished with all computations, please write your final answer value without any units, using the following formats in the boxed form \\boxed{{}}:

- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)"""

MEDCALC = "ncbi/MedCalc-Bench"
NGRAM = 16
TRAIN_CAP = 10000
END_OF_TURN = "<|im_end|>"


def options_block(question, options):
    return f"{question}\n" + "; ".join(f"{k} {options[k]}" for k in sorted(options))


def row(source, split, question, gt, calc=False, **extra):
    template = CALC_BASE_TEMPLATE if calc else QA_BASE_TEMPLATE
    return dict(source=source, split=split, question=question, gt=str(gt),
                user=template.format(task_description=question), **extra)


def medqa(split_rows, split):
    out = []
    for ex in split_rows:
        raw = ex["data"] if isinstance(ex["data"], dict) else ast.literal_eval(ex["data"])
        out.append(row("openlifescienceai/medqa", split,
                       options_block(raw["Question"], raw["Options"]), raw["Correct Option"]))
    return out


def medmcqa(split_rows, split):
    letters = "ABCD"
    return [row("openlifescienceai/medmcqa", split,
                options_block(ex["question"], dict(zip(letters, (ex["opa"], ex["opb"], ex["opc"], ex["opd"])))),
                letters[int(ex["cop"])]) for ex in split_rows]


def headqa(zip_path, which, split):
    """Same traversal as the HF head_qa.py script (config "en"), keeping only
    questions without an image, as FaithMed's filter_headqa_no_image does."""
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read(f"HEAD_EN/{which}_HEAD_EN.json").decode("utf-8"))
    out = []
    for exam in data["exams"]:
        for q in data["exams"][exam]["data"]:
            if q["image"].strip():
                continue
            answers = sorted(q["answers"], key=lambda a: int(a["aid"]))
            options, gt = {}, None
            for i, a in enumerate(answers):
                letter = chr(ord("A") + i)
                options[letter] = a["atext"].strip()
                if str(a["aid"]).strip() == q["ra"].strip():
                    gt = letter
            out.append(row("dvilares/head_qa", split, options_block(q["qtext"].strip(), options), gt))
    return out


def medcalc(split_rows, split):
    return [row(MEDCALC, split, f"Patient Note:\n{ex['Patient Note']}\n\nQuestion:\n{ex['Question']}",
                ex["Ground Truth Answer"], calc=True, lower=ex["Lower Limit"], upper=ex["Upper Limit"],
                calc_id=str(ex["Calculator ID"]), output_type=ex["Output Type"]) for ex in split_rows]


def medbullets(split_rows, split):
    return [row("JesseLiu/medbulltes5op", split,
                options_block(ex["question"], {k: ex[f"choices{k}"] for k in "ABCDE"}),
                str(ex["answer_idx"]).strip().upper()) for ex in split_rows]


def mmlu_pro_health(split_rows, split):
    return [row("TIGER-Lab/MMLU-Pro", split,
                options_block(ex["question"], {chr(ord("A") + i): t for i, t in enumerate(ex["options"])}),
                str(ex["answer"]).strip().upper())
            for ex in split_rows if (ex.get("category") or "").lower() == "health"]


def medxpertqa(split_rows, split):
    return [row("TsinghuaC3I/MedXpertQA", split, ex["question"], ex["label"]) for ex in split_rows]


def load_all(cache_dir):
    """Train and test rows in FaithMed's dataset_specs order."""
    load = lambda name, cfg=None: datasets.load_dataset(name, cfg, cache_dir=cache_dir)
    train, test = [], []

    d = load("openlifescienceai/medqa")
    train += medqa(d["train"], "train") + medqa(d["dev"], "train")
    test += medqa(d["test"], "test")

    z = hf_hub_download("dvilares/head_qa", "data/head-qa-es-en-pdfs.zip", repo_type="dataset", cache_dir=cache_dir)
    train += headqa(z, "train", "train") + headqa(z, "dev", "train")
    test += headqa(z, "test", "test")

    d = load("openlifescienceai/medmcqa")
    train += medmcqa(d["train"], "train")
    test += medmcqa(d["validation"], "validation")

    d = load(MEDCALC)
    train += medcalc(d["train"], "train")
    test += medcalc(d["test"], "test")

    test += medbullets(load("JesseLiu/medbulltes5op")["test"], "test")
    test += mmlu_pro_health(load("TIGER-Lab/MMLU-Pro")["test"], "test")
    test += medxpertqa(load("TsinghuaC3I/MedXpertQA", "Text")["test"], "test")
    return train, test


def norm_words(text):
    return " ".join(str(text).lower().split()).split()


def grams(text):
    w = norm_words(text)
    return {" ".join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/medmix_sft")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--cache_dir", default=None, help="HF cache (default: HF_HOME)")
    args = parser.parse_args()

    train, test = load_all(args.cache_dir)
    n_in = collections.Counter(r["source"] for r in train)

    # Same shuffle as FaithMed: datasets.shuffle(seed=42) on the concatenation.
    order = datasets.Dataset.from_dict({"i": list(range(len(train)))}).shuffle(seed=42)["i"]
    train = [train[i] for i in order]
    t_order = datasets.Dataset.from_dict({"i": list(range(len(test)))}).shuffle(seed=42)["i"]
    test = [test[i] for i in t_order]

    test_grams = set()
    for r in test:
        if r["source"] != MEDCALC:
            test_grams |= grams(r["question"])
    kept = [r for r in train if r["source"] == MEDCALC or not (grams(r["question"]) & test_grams)]
    n_decon = collections.Counter(r["source"] for r in kept)

    counts, capped = collections.Counter(), []
    for r in kept:
        if counts[r["source"]] < TRAIN_CAP:
            capped.append(r)
            counts[r["source"]] += 1

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def prompt(r):
        return tok.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    # A random sample, not the first rows: the shuffled stream starts MedMCQA-heavy
    # (it is 87% of the rows before the cap), so a prefix would not match the mix.
    val_idx = set(random.Random(42).sample(range(len(capped)), args.val_size))
    val = [r for i, r in enumerate(capped) if i in val_idx]
    train_out = [r for i, r in enumerate(capped) if i not in val_idx]
    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("train", train_out), ("val", val)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"prompt": prompt(r), "answer": f"\\boxed{{{r['gt']}}}{END_OF_TURN}",
                                    "source": r["source"]}) + "\n")
    with open(os.path.join(args.out, "test.jsonl"), "w") as f:
        for r in test:
            out = {k: v for k, v in r.items() if k not in ("user", "split")}
            f.write(json.dumps({**out, "prompt": prompt(r)}) + "\n")

    lengths = sorted(len(tok(prompt(r), add_special_tokens=False)["input_ids"]) for r in train_out)
    report = {
        "train_in_by_source": n_in,
        "after_16gram_by_source": n_decon,
        "after_cap_by_source": counts,
        "train_out": len(train_out), "val_out": len(val),
        "test_by_source": collections.Counter(r["source"] for r in test),
        "train_prompt_tokens": {"mean": round(sum(lengths) / len(lengths)), "p99": lengths[int(.99 * len(lengths))],
                                "max": lengths[-1], "over_1024": sum(l > 1024 for l in lengths),
                                "over_2048": sum(l > 2048 for l in lengths)},
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
