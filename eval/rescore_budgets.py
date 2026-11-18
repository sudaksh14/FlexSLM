"""Accuracy at smaller decoding budgets from a finished eval_gen run, offline (no GPU).

A greedy or seeded-sampled trajectory cut at b tokens IS the run with max_new_tokens=b, so the stored
generations (<name>.samples.jsonl) of one long run give the whole accuracy-vs-budget curve. Needed when
--budgets was not passed (job 28396: sbatch --export splits BUDGETS=64,128,... at the commas).
Rows are re-joined with the test file (same --sources / --per_source filter, same order), because MedCalc-Bench
scoring needs the gold limits that the samples file does not store.

    python -m eval.rescore_budgets results/gen.<model>.t4096.pthink.n300.sq3 --budgets 64,128,256,512,1024,2048,4096 \
        --tokenizer Qwen/Qwen3-1.7B --per_source 300
"""
import argparse
import collections
import json

from eval.eval_gen import is_correct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stem", help="results path without .json / .samples.jsonl")
    p.add_argument("--budgets", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", default="data/medmix_sft/test.jsonl")
    p.add_argument("--per_source", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = [json.loads(line) for line in open(args.data)]
    if args.per_source:
        seen = collections.Counter()
        rows = [r for r in rows if (seen.update([r["source"]]) or seen[r["source"]] <= args.per_source)]
    samples = [json.loads(line) for line in open(args.stem + ".samples.jsonl")]
    assert len(rows) == len(samples), (len(rows), len(samples))
    assert all(r["source"] == s["source"] and r["gt"] == s["gt"] for r, s in zip(rows, samples))
    ids = [tok(s["gen"], add_special_tokens=False)["input_ids"] for s in samples]

    out = {}
    for b in sorted({int(x) for x in args.budgets.split(",")}):
        per = collections.defaultdict(list)
        toks = []
        for r, s, i in zip(rows, samples, ids):
            text = s["gen"] if len(i) <= b else tok.decode(i[:b], skip_special_tokens=True)
            per[r["source"]].append(is_correct(r, text))
            toks.append(min(s["gen_tokens"], b))
        accs = {src: sum(v) / len(v) for src, v in sorted(per.items())}
        accs["avg"] = sum(accs.values()) / len(accs)
        accs["mean_gen_tokens"] = sum(toks) / len(toks)
        out[b] = accs
        print(f"budget {b:5d}: avg {100 * accs['avg']:5.1f}  mean generated tokens {accs['mean_gen_tokens']:7.1f}")
    json.dump({"stem": args.stem, "per_source": args.per_source, "budgets": out}, open(args.out or args.stem + ".budgets.json", "w"), indent=2)


if __name__ == "__main__":
    main()
