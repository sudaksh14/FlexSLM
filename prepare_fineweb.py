"""
One-off tokenization of FineWeb-Edu into flat binary token files, for the memmap
data pipeline (utils.load_fineweb_edu_memmap).

Streams HuggingFaceFW/fineweb-edu sample-10BT once, tokenizes with the requested
tokenizer, and appends token ids to train.bin / val.bin. Training then reads
random windows out of those via np.memmap, which removes the per-epoch
re-tokenization cost of the streaming loader and gives true random shuffling with
flat (page-cache backed) memory.

The output is tokenizer-specific, so run this once per tokenizer you train with.
Token width is chosen from the vocab size: uint16 when vocab < 65536 (llama,
smollm2, gpt2), uint32 otherwise (qwen2.5's 151936 does not fit in uint16).

Run via prepare_fineweb.sh (sbatch), not directly on the login node.

  python3 prepare_fineweb.py --tokenizer JackFram/llama-160m --max-tokens 1_400_000_000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import config.paths as paths
from utils import fineweb_bin_dir

WRITE_CHUNK_TOKENS = 50_000_000  # flush to disk about every 50M tokens


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer name, e.g. JackFram/llama-160m or Qwen/Qwen2.5-1.5B")
    p.add_argument("--max-tokens", type=lambda s: int(float(s.replace("_", ""))), default=1_400_000_000,
                   help="total training tokens to write (default 1.4e9)")
    p.add_argument("--val-tokens", type=lambda s: int(float(s.replace("_", ""))), default=2_000_000,
                   help="tokens held out for validation, taken from the front of the stream")
    p.add_argument("--sample", default="sample-10BT",
                   choices=["sample-10BT", "sample-100BT", "sample-350BT"],
                   help="FineWeb-Edu subset. sample-10BT caps out at 10B tokens, which is "
                        "below the compute-optimal horizon for the 1.5B/1.7B teachers - "
                        "use sample-100BT for those.")
    p.add_argument("--batch-docs", type=int, default=1000, help="documents per tokenizer call")
    p.add_argument("--out-dir", default=None, help="override output directory")
    p.add_argument("--overwrite", action="store_true", help="rebuild even if outputs already exist")
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = args.out_dir or fineweb_bin_dir(args.tokenizer)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")

    if os.path.exists(meta_path) and not args.overwrite:
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Already prepared at {out_dir}: {meta}\nPass --overwrite to rebuild.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    # uint16 tops out at 65535, which is below Qwen2.5's 151936 - silently
    # truncating there would corrupt every token id above the limit.
    dtype = np.uint16 if vocab_size < 2 ** 16 else np.uint32
    print(f"tokenizer={args.tokenizer} vocab_size={vocab_size} -> dtype={np.dtype(dtype).name} "
          f"({np.dtype(dtype).itemsize} bytes/token)", flush=True)
    print(f"target: {args.val_tokens:,} val + {args.max_tokens:,} train tokens", flush=True)
    print(f"source: HuggingFaceFW/fineweb-edu:{args.sample}", flush=True)
    print(f"output: {out_dir}", flush=True)

    stream = load_dataset("HuggingFaceFW/fineweb-edu", name=args.sample,
                          split="train", streaming=True)

    targets = {"val": args.val_tokens, "train": args.max_tokens}
    written = {"val": 0, "train": 0}
    docs_seen = 0
    t0 = time.time()

    doc_iter = iter(stream)
    for split in ("val", "train"):  # val first so it is a disjoint prefix of the stream
        path = os.path.join(out_dir, f"{split}.bin")
        target = targets[split]
        buf = []
        buf_len = 0
        with open(path, "wb") as fh:
            while written[split] < target:
                batch = []
                for _ in range(args.batch_docs):
                    try:
                        batch.append(next(doc_iter)["text"])
                    except StopIteration:
                        break
                if not batch:
                    print(f"WARNING: stream exhausted with {written[split]:,}/{target:,} "
                          f"{split} tokens written", flush=True)
                    break
                docs_seen += len(batch)

                for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                    buf.append(np.asarray(ids, dtype=dtype))
                    buf_len += len(ids)

                if buf_len >= WRITE_CHUNK_TOKENS:
                    arr = np.concatenate(buf)
                    keep = min(len(arr), target - written[split])
                    fh.write(arr[:keep].tobytes())
                    written[split] += keep
                    buf, buf_len = [], 0
                    pct = 100 * written[split] / target
                    rate = written[split] / max(1e-9, time.time() - t0)
                    print(f"  {split}: {written[split]:,}/{target:,} tokens ({pct:.1f}%) "
                          f"{docs_seen:,} docs  {rate/1e6:.2f}M tok/s", flush=True)

            if buf and written[split] < target:
                arr = np.concatenate(buf)
                keep = min(len(arr), target - written[split])
                fh.write(arr[:keep].tobytes())
                written[split] += keep
        print(f"{split}.bin done: {written[split]:,} tokens "
              f"({os.path.getsize(path)/2**30:.2f} GiB)", flush=True)

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": written["train"],
        "val_tokens": written["val"],
        "docs_consumed": docs_seen,
        "source": f"HuggingFaceFW/fineweb-edu:{args.sample}",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {meta_path}\n{json.dumps(meta, indent=2)}")
    print(f"total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
    # The HF fast tokenizer's Rust threads can abort during interpreter shutdown
    # ("PyGILState_Release: thread state must be current when releasing"), which
    # makes SLURM mark the job FAILED even though every byte was already written
    # and flushed above. Everything is on disk by here, so skip finalization and
    # exit cleanly - otherwise --dependency=afterok chains would break on a
    # successful run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
