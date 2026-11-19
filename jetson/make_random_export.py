#!/usr/bin/env python3
"""Random-weight MedNest export for SHAPE-ONLY Jetson runs (latency, memory, energy, switch cost).

Writes the layout of training/export_flex_hf.py: <out>/level<i>/config.json (+ tokenizer) for every
width, and bf16 safetensors for the widest level and for the dense anchor levels. Weights are random
(seeded), and every anchor is the exact prefix slice of the full one, as in a real export, so
views, slice loading and reloading behave as with trained weights. Answers are meaningless: run the
benchmarks with MEDNEST_FORCE_LEN=1 and never report accuracy from such an export.

    python3 jetson/make_random_export.py --source ~/mednest/models/Qwen3-0.6B --out ~/mednest/models/mednest-random

Widths = the final 9-width MedNest grid of Qwen3-0.6B (training/stage1.FRACTIONS_9, level 8 = the
full model); override with --widths-json (list of {hidden_size, intermediate_size,
num_attention_heads, num_key_value_heads}).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# config.experiments s1.qwen3_0p6b.full (hidden, heads, kv heads, mlp), head_dim 128
HIDDEN = (512, 576, 640, 704, 768, 832, 896, 960, 1024)
HEADS = (8, 8, 10, 12, 12, 12, 14, 16, 16)
KV = (4, 4, 5, 6, 6, 6, 7, 8, 8)
MLP = (1536, 1728, 1920, 2112, 2304, 2496, 2688, 2880, 3072)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="dir of Qwen3-0.6B (config + tokenizer)")
    p.add_argument("--out", required=True)
    p.add_argument("--anchors", type=int, nargs="*", default=[0, 4], help="dense levels whose weights are written")
    p.add_argument("--widths-json", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from mednest_jetson.elastic import _prefix

    if args.widths_json:
        widths = json.load(open(args.widths_json))
    else:
        widths = [dict(hidden_size=h, num_attention_heads=a, num_key_value_heads=k, intermediate_size=m)
                  for h, a, k, m in zip(HIDDEN, HEADS, KV, MLP)]
    src = AutoConfig.from_pretrained(args.source)
    top = len(widths) - 1
    cfgs = []
    for w in widths:
        c = src.__class__.from_dict(src.to_dict())
        for k, v in w.items():
            setattr(c, k, v)
        if hasattr(c, "head_dim"):
            c.head_dim = getattr(src, "head_dim", None) or w["hidden_size"] // w["num_attention_heads"]
        cfgs.append(c)
    for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"):
        if getattr(cfgs[top], k) != getattr(src, k):
            raise SystemExit(f"widest level differs from the source model in {k}")

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    full = AutoModelForCausalLM.from_config(cfgs[top], torch_dtype=torch.bfloat16).eval()
    full_sd = full.state_dict()
    tok_files = [f for f in os.listdir(args.source)
                 if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt", "special_tokens_map.json",
                                                       "added_tokens.json", "generation_config.json")]
    for level, cfg in enumerate(cfgs):
        d = os.path.join(args.out, f"level{level}")
        os.makedirs(d, exist_ok=True)
        if level == top or level in args.anchors:
            m = full if level == top else AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).eval()
            if level != top:
                m.load_state_dict({k: _prefix(full_sd[k], v.shape).clone() for k, v in m.state_dict().items()})
            m.save_pretrained(d, safe_serialization=True)      # also writes config.json
            if level != top:
                del m
        else:
            cfg.save_pretrained(d)
        for f in tok_files:
            shutil.copy2(os.path.join(args.source, f), os.path.join(d, f))
        print(f"level{level}: hidden {cfg.hidden_size} heads {cfg.num_attention_heads}/{cfg.num_key_value_heads} "
              f"mlp {cfg.intermediate_size}" + ("  [weights]" if (level == top or level in args.anchors) else ""))
    with open(os.path.join(args.out, "RANDOM_INIT.txt"), "w") as f:
        f.write("Random weights (seed %d): shape-only measurements, no accuracy.\n" % args.seed)


if __name__ == "__main__":
    main()
