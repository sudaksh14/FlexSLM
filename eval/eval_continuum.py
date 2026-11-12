"""
Continuum-of-widths evaluation: PMC val perplexity of sub-networks of a trained
FlexLLaMA at widths it may never have trained, plus plain Hugging Face models.

Methods compared (implementation_log.md, 2026-11-12):
  1. zero-shot interpolation: uniform prefix widths on the 1/16 grid (0.25 ... 1.0),
     any checkpoint (also shows how far below the smallest trained width it holds up);
  2. anchors + random grid widths in training (s1.*.pilot.grid9_anchors_plus1): same
     uniform evaluation, the in-between widths are trained;
  3. per-width norm offsets / LLaMaFlex modulation (pilots level_norms, mod_*): same
     uniform evaluation; offsets are interpolated, modulation is width-conditioned;
  4. Mix'n'Match (MatFormer): one residual width for the whole model, per layer the MLP
     width and head count of a trained anchor (k layers at the bigger anchor, the rest
     at the smaller), giving budgets between two anchors without any training.

Windows are the Stage-1 validation windows (val.bin, seed 1338, 640 x 1,024), so the
trained anchors reproduce the pilots' val_level*_ppl (a built-in check).

  python -m eval.eval_continuum --config s1.qwen3_0p6b.pilot.base --out results/continuum.base.json
  python -m eval.eval_continuum --config s1.qwen3_0p6b.pilot.base --init ...   # untrained warm start
  python -m eval.eval_continuum --hf Qwen/Qwen3-1.7B --out results/pmcppl.qwen3-1.7b.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

PMC_DIR = os.path.join("data", "pmc_qwen3")
VAL_SEED = 1338  # load_pmc_memmap: seed 1337 + 1 for val


def val_batches(data_dir: str, windows: int, seq_len: int, batch_size: int):
    import utils
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    ds = utils._MemmapWindowDataset(os.path.join(data_dir, "val.bin"), meta["dtype"], seq_len, windows,
                                    seed=VAL_SEED)
    items = [ds[i] for i in range(len(ds))]
    return [torch.stack(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]


@torch.no_grad()
def perplexity(forward, batches, device) -> float:
    nll, n = 0.0, 0
    for b in batches:
        b = b.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = forward(b)
        nll += F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), b[:, 1:].flatten(), reduction="sum").item()
        n += b[:, 1:].numel()
    return math.exp(nll / n)


def uniform_dims(cfg, frac: float) -> dict:
    """Widths of fraction `frac`, rounded exactly like the level builder
    (config.experiments._flex_llama_dims_fixed_head): hidden / MLP to multiples of 64,
    heads to multiples of the GQA ratio."""
    H, I = list(cfg.hidden_dims)[-1], list(cfg.intermediate_dims)[-1]
    nh, nkv = list(cfg.num_heads)[-1], list(cfg.num_kv_heads)[-1]
    if frac >= 1:
        return dict(hidden=H, intermediate=I, heads=nh)
    ratio = nh // nkv
    return dict(hidden=min(H, max(64, round(H * frac / 64) * 64)),
                intermediate=min(I, max(64, round(I * frac / 64) * 64)),
                heads=min(nh, max(ratio, round(nh * frac / ratio) * ratio)))


def count_params(cfg, hidden: int, inter: list, heads: list) -> tuple[int, int]:
    """(total, non-embedding) parameters of a sub-network (tied embeddings)."""
    hd = cfg.head_dim or list(cfg.hidden_dims)[-1] // list(cfg.num_heads)[-1]
    ratio = list(cfg.num_heads)[-1] // list(cfg.num_kv_heads)[-1]
    body = hidden  # final norm
    for m, nq in zip(inter, heads):
        nkv = nq // ratio
        body += hidden * nq * hd * 2 + hidden * nkv * hd * 2  # q, o, k, v
        body += (2 * hd if cfg.qk_norm else 0) + 3 * hidden * m + 2 * hidden
    emb = cfg.vocab_size * hidden
    return emb + body, body


def mixnmatch_candidates(cfg, anchors, ks=(7, 14, 21)):
    """Between each pair of adjacent anchors (lo, hi): residual width of hi with k layers'
    blocks shrunk to lo, or residual width of lo with k layers' blocks grown to hi;
    the k layers are the first, the last, or spread evenly."""
    L = cfg.num_layers
    out = []
    for f_lo, f_hi in zip(anchors, anchors[1:]):
        lo, hi = uniform_dims(cfg, f_lo), uniform_dims(cfg, f_hi)
        for base, other, tag in ((hi, lo, "hidden_hi"), (lo, hi, "hidden_lo")):
            for k in ks:
                for place in ("first", "last", "spread"):
                    if place == "first":
                        sel = set(range(k))
                    elif place == "last":
                        sel = set(range(L - k, L))
                    else:
                        sel = {round(i * L / k) for i in range(k)}
                    inter = [other["intermediate"] if i in sel else base["intermediate"] for i in range(L)]
                    heads = [other["heads"] if i in sel else base["heads"] for i in range(L)]
                    out.append(dict(kind="mixnmatch", desc=f"{f_lo}-{f_hi} {tag} k{k} {place}",
                                    hidden=base["hidden"], intermediate=inter, heads=heads))
    return out


def load_flex(config: str, init: bool, device: str):
    import utils
    from training.run_experiment import resolve_from_str
    cfg = resolve_from_str(f"flexllama,{config}").model_config
    if init:
        model = cfg.make_model()
        utils.load_llama_weights_into_flexllama(model, cfg.pretrained_hf_model)
    else:
        model = utils.load_model(f"flexllama,{config}", cfg)
    return cfg, model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="trained FlexLLaMA config (saved_models/<config>.pt)")
    src.add_argument("--hf", help="plain Hugging Face model: perplexity only")
    p.add_argument("--init", action="store_true", help="evaluate the untrained warm start of --config")
    p.add_argument("--fractions", default=",".join(str(0.25 + k / 16) for k in range(13)),
                   help="uniform widths (default 0.25, 0.3125, ..., 1.0)")
    p.add_argument("--anchors", default="0.5,0.75,1.0", help="trained deployment widths for Mix'n'Match")
    p.add_argument("--no-mixnmatch", action="store_true")
    p.add_argument("--windows", type=int, default=640)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--data-dir", default=PMC_DIR)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batches = val_batches(args.data_dir, args.windows, args.seq_len, args.batch_size)
    t0 = time.time()
    rows = []

    if args.hf:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.bfloat16).to(device).eval()
        ppl = perplexity(lambda b: model(input_ids=b, use_cache=False).logits, batches, device)
        rows.append(dict(kind="hf", desc=args.hf, params=sum(p.numel() for p in model.parameters()), ppl=ppl))
        print(f"{args.hf}: PMC val ppl {ppl:.3f}", flush=True)
    else:
        cfg, model = load_flex(args.config, args.init, device)
        trained = [dict(hidden=h, intermediate=m, heads=n) for h, m, n in
                   zip(cfg.hidden_dims, cfg.intermediate_dims, cfg.num_heads)]
        L = cfg.num_layers
        # Trained levels through set_level_use (reference) ...
        for lvl, d in enumerate(trained):
            model.set_level_use(lvl)
            ppl = perplexity(model, batches, device)
            tot, body = count_params(cfg, d["hidden"], [d["intermediate"]] * L, [d["heads"]] * L)
            rows.append(dict(kind="level", desc=f"level{lvl}", **d, params=tot, params_nonemb=body, ppl=ppl))
            print(f"level {lvl} {d}: {tot / 1e6:.0f}M ppl {ppl:.3f}", flush=True)
        # ... then every width through set_widths.
        cands = []
        for f in (float(x) for x in args.fractions.split(",")):
            d = uniform_dims(cfg, f)
            is_trained = any(d == t for t in trained)
            cands.append(dict(kind="uniform", desc=f"{f:.4f}" + (" (trained)" if is_trained else ""),
                              hidden=d["hidden"], intermediate=[d["intermediate"]] * L, heads=[d["heads"]] * L))
        if not args.no_mixnmatch:
            cands += mixnmatch_candidates(cfg, [float(x) for x in args.anchors.split(",")])
        for c in cands:
            model.set_widths(c["hidden"], c["intermediate"], c["heads"])
            ppl = perplexity(model, batches, device)
            tot, body = count_params(cfg, c["hidden"], c["intermediate"], c["heads"])
            rows.append(dict(c, params=tot, params_nonemb=body, ppl=ppl))
            print(f"{c['kind']:9s} {c['desc']:32s} hidden {c['hidden']:4d} {tot / 1e6:6.0f}M ppl {ppl:.3f}", flush=True)
        model.set_level_use(model.max_level())

    out = dict(source=args.hf or args.config, init=args.init, windows=args.windows, seq_len=args.seq_len,
               seconds=round(time.time() - t0), rows=rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
