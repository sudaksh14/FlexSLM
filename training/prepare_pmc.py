"""
One-off tokenization of PMC Open Access (commercial-use subset, the BioMistral
corpus) into flat token files for Stage 1 (training/stage1.py: load_pmc_memmap).

Articles are fetched from the public S3 bucket by PMCID (pmc_commercial_ids.txt,
shuffled with the same seed as utils.load_pmc_open_access), in a fixed order with
retries; documents are joined with <|endoftext|>. val.bin, then test.bin, then
train.bin are filled from consecutive, disjoint runs of documents, so no test or
validation article is ever trained on. meta.json records counts, the dtype and
the number of articles that could not be fetched.

CPU job on DAS-6 (fatq), never on the login node:
  sbatch scripts/das6/prepare_pmc.sbatch [train_tokens]
  python -m training.prepare_pmc --train-tokens 1e9 --out-dir data/pmc_qwen3
"""
import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import urllib.request

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # common.sh disables it; this job wants all cores

import numpy as np

_num = lambda s: int(float(str(s).replace("_", "")))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--train-tokens", type=_num, default=1_000_000_000)
    p.add_argument("--val-tokens", type=_num, default=2_000_000)
    p.add_argument("--test-tokens", type=_num, default=2_000_000)
    p.add_argument("--ids-file", default="pmc_commercial_ids.txt")
    p.add_argument("--out-dir", default=os.path.join("data", "pmc_qwen3"))
    p.add_argument("--workers", type=int, default=64, help="concurrent downloads")
    p.add_argument("--block", type=int, default=512, help="articles fetched and tokenized per block")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fetch_text(pmcid: str, retries: int = 3):
    url = f"https://pmc-oa-opendata.s3.amazonaws.com/{pmcid}.1/{pmcid}.1.txt"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")
    if os.path.exists(meta_path) and not args.overwrite:
        print(f"Already prepared: {open(meta_path).read()}\nPass --overwrite to rebuild.")
        return

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sep = tok.convert_tokens_to_ids("<|endoftext|>")
    if sep is None or sep == tok.unk_token_id:
        sep = tok.eos_token_id
    dtype = np.uint16 if len(tok) < 2 ** 16 else np.uint32

    with open(args.ids_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    random.Random(42).shuffle(ids)  # same order as utils.load_pmc_open_access
    print(f"{len(ids):,} PMCIDs, tokenizer {args.tokenizer} (vocab {len(tok)}, {np.dtype(dtype).name}), "
          f"separator id {sep}, targets val {args.val_tokens:,} test {args.test_tokens:,} "
          f"train {args.train_tokens:,}", flush=True)

    splits = [("val", args.val_tokens), ("test", args.test_tokens), ("train", args.train_tokens)]
    written = {name: 0 for name, _ in splits}
    docs = {name: 0 for name, _ in splits}
    failed = 0
    pos = 0  # next PMCID to fetch
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, target in splits:
            with open(os.path.join(args.out_dir, f"{name}.bin"), "wb") as fh:
                last_print = 0
                while written[name] < target:
                    if pos >= len(ids):
                        print(f"WARNING: ran out of articles, {name} has {written[name]:,}/{target:,}", flush=True)
                        break
                    block = ids[pos:pos + args.block]
                    pos += len(block)
                    texts = list(pool.map(fetch_text, block))  # order preserved
                    good = [t for t in texts if t]
                    failed += len(texts) - len(good)
                    if not good:
                        continue
                    for seq in tok(good, add_special_tokens=False)["input_ids"]:
                        if written[name] >= target:
                            break
                        arr = np.asarray(seq + [sep], dtype=dtype)[: target - written[name]]
                        fh.write(arr.tobytes())
                        written[name] += len(arr)
                        docs[name] += 1
                    if written[name] - last_print >= 50_000_000 or written[name] >= target:
                        last_print = written[name]
                        rate = sum(written.values()) / max(1e-9, time.time() - t0)
                        print(f"  {name}: {written[name]:,}/{target:,} tokens, {docs[name]:,} docs, "
                              f"{pos:,} ids read, {failed:,} failed, {rate / 1e6:.2f}M tok/s", flush=True)

    meta = dict(tokenizer=args.tokenizer, vocab_size=len(tok), dtype=np.dtype(dtype).name,
                separator_id=int(sep), train_tokens=written["train"], val_tokens=written["val"],
                test_tokens=written["test"], docs=docs, ids_read=pos, fetch_failed=failed,
                source="PMC OA commercial-use subset, s3://pmc-oa-opendata, ids shuffled with Random(42)",
                seconds=round(time.time() - t0))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    if written["train"] < args.train_tokens:
        sys.exit(1)


if __name__ == "__main__":
    main()
