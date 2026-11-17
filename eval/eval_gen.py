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
Other model families (--prompt_format): the user message is taken out of the stored Qwen3
prompt (same instruction and question) and rendered in the model's own format, e.g. Apollo's
"User:{query}\nAssistant:" (FreedomIntelligence model cards) or the tokenizer's chat template.

    python3 -m eval.eval_gen hf_export/<config>/level2 --out results/gen.<name>.level2.json
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


QWEN3_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
PROMPT_FORMATS = ("as_is", "apollo", "chat", "apollo_answer")
# apollo_answer: Apollo's own evaluation style (arXiv 2403.03640, Sect. 4): the assistant turn is
# pre-filled with "The correct answer is" and the answer is read from the first tokens. Our
# instruction asks for step-by-step reasoning and a \boxed{} answer, which these models do not
# produce (0% boxed with 1,024 tokens, jobs 28309/28310), so it is replaced by a direct request.
MC_INSTRUCTION = r"Let's think step by step and include the final choice in the box \boxed{}."
MC_DIRECT = "Answer with the letter of the correct option."
CALC_INSTRUCTION = r"using the following formats in the boxed form \boxed{}:"
CALC_DIRECT = "using the following formats:"
ANSWER_PREFIX = {"mc": "The correct answer is", "calc": "The answer is"}
FIRST_LETTER = re.compile(r"^[\s:(\"'*]*([A-J])\b")
FIRST_VALUE = re.compile(r"\(\s*\d+\s*weeks?\s*,\s*\d+\s*days?\s*\)|\d{1,2}/\d{1,2}/\d{2,4}|-?\d[\d,]*(?:\.\d+)?")


def render_prompt(prompt: str, fmt: str, tokenizer=None) -> str:
    """as_is: the stored (Qwen3, thinking disabled) prompt; apollo: Apollo/Apollo2's
    "User:{query}\\nAssistant:"; apollo_answer: the same with a direct-answer instruction and the
    assistant turn pre-filled; chat: the tokenizer's own chat template."""
    if fmt == "as_is":
        return prompt
    m = QWEN3_USER.search(prompt)
    if m is None:
        raise ValueError("cannot find the user message in the stored prompt")
    query = m.group(1)
    if fmt == "apollo":
        return f"User:{query}\nAssistant:"
    if fmt == "apollo_answer":
        calc = CALC_INSTRUCTION in query
        new = query.replace(CALC_INSTRUCTION, CALC_DIRECT) if calc else query.replace(MC_INSTRUCTION, MC_DIRECT)
        if new == query:
            raise ValueError("unknown instruction in the stored prompt")
        return f"User:{new}\nAssistant:{ANSWER_PREFIX['calc' if calc else 'mc']}"
    if fmt == "chat":
        return tokenizer.apply_chat_template([{"role": "user", "content": query}], tokenize=False,
                                             add_generation_prompt=True)
    raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}, got {fmt!r}")


def forced_answer(gen: str, row: dict):
    r"""apollo_answer: the answer right after the pre-filled prefix, as a \boxed{} string for the
    usual scoring (None if there is none). A \boxed{} the model wrote itself takes precedence."""
    if last_boxed(gen) is not None:
        return gen
    if row["source"] == MEDCALC:
        m = FIRST_VALUE.search(gen[:40])
        return None if m is None else r"\boxed{" + m.group(0) + "}"
    m = FIRST_LETTER.search(gen)
    return None if m is None else r"\boxed{" + m.group(1) + "}"


def is_correct(row: dict, gen: str) -> bool:
    return score_medcalc(gen, row) if row["source"] == MEDCALC else score_choice(gen, row["gt"])


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size: int, max_new_tokens: int, add_special_tokens: bool = False):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=add_special_tokens).to(model.device)
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
    parser.add_argument("--prompt_format", default="as_is", choices=PROMPT_FORMATS)
    # Only for models whose architecture is not in transformers (Apollo-1.8B = Qwen-1 code); always
    # together with --revision pinned to a commit whose code was checked.
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.trust_remote_code and not args.revision:
        parser.error("--trust_remote_code needs --revision (a pinned, reviewed commit)")
    extra = dict(trust_remote_code=True, revision=args.revision) if args.trust_remote_code else (
        dict(revision=args.revision) if args.revision else dict())
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", **extra)
    if tokenizer.pad_token is None:
        # Qwen-1 tokenizers (Apollo-1.8B) define neither; their generation config pads with <|endoftext|>.
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **extra).to(device).eval()

    rows = [json.loads(line) for line in open(args.data)]
    if args.sources:
        keep = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
    prompts = [render_prompt(r["prompt"], args.prompt_format, tokenizer) for r in rows]
    if args.prompt_format != "as_is":
        print(f"[prompt {args.prompt_format}] {prompts[0]!r}")
    # The stored prompts and chat templates carry their own special tokens; the plain Apollo
    # format needs the model's BOS (Gemma-based Apollo models expect <bos>), added by the tokenizer.
    gens = generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens,
                    add_special_tokens=args.prompt_format.startswith("apollo"))

    per = collections.defaultdict(list)
    samples = []
    for r, g in zip(rows, gens):
        scored = (forced_answer(g, r) or "") if args.prompt_format == "apollo_answer" else g
        ok = is_correct(r, scored)
        per[r["source"]].append(ok)
        samples.append({"source": r["source"], "gt": r["gt"], "boxed": last_boxed(scored), "correct": ok, "gen": g})

    results = {}
    for src, oks in sorted(per.items()):
        acc = sum(oks) / len(oks)
        results[src] = {"acc": acc, "stderr": math.sqrt(acc * (1 - acc) / len(oks)), "n": len(oks),
                        "no_boxed": sum(s["boxed"] is None for s in samples if s["source"] == src)}
        print(f"{src:28s} acc {100 * acc:5.1f} ± {100 * results[src]['stderr']:.1f}  "
              f"(n {len(oks)}, no \\boxed {results[src]['no_boxed']})")
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "max_new_tokens": args.max_new_tokens,
                   "prompt_format": args.prompt_format, "results": results}, f, indent=2)
    with open(args.out.replace(".json", ".samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
