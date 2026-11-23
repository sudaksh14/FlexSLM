"""Weight-only int8 (symmetric, one scale per output row, max-abs / 127; embedding, output head and norms stay bf16) written back
as bf16 weights, so eval/eval_gen.py evaluates exactly the numbers an int8 layer computes (weight = int8 * row scale, in bf16).

    python -m eval.fake_quant FULL_DIR OUT_FULL [--slice NARROW_DIR OUT_NARROW ...]

--slice: a narrower level of the SAME nested model (its weights must be exact prefix slices of FULL_DIR; checked). It gets the
FULL model's row scales and int8 weights sliced, i.e. what one shared int8 copy on the device gives, not its own quantisation.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def quantise(w):
    """(int8 weight, bf16 row scale) of a bf16 weight matrix [out, in]."""
    s = (w.float().abs().amax(1, keepdim=True) / 127).clamp_min(1e-30).to(torch.bfloat16)
    q = torch.round(w.float() / s.float()).clamp_(-127, 127).to(torch.int8)
    return q, s


def dequantise(q, s):
    return (q.to(torch.bfloat16) * s).to(torch.bfloat16)


def is_linear(name):
    return name.endswith("_proj.weight")       # q,k,v,o and gate,up,down; not embed_tokens / lm_head / norms


def load(d):
    with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def write(src, out, tensors):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f != "model.safetensors":
            shutil.copy(os.path.join(src, f), os.path.join(out, f))
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"scheme": "fake int8 (weight-only, symmetric, per output row, embedding/lm_head bf16)"}, open(os.path.join(out, "FAKE_INT8.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("out")
    ap.add_argument("--slice", nargs="*", default=[], help="pairs: NARROW_DIR OUT_DIR ...")
    a = ap.parse_args()
    full = load(a.full)
    qs = {k: quantise(v) for k, v in full.items() if is_linear(k)}
    err = max(float((dequantise(*qs[k]).float() - full[k].float()).abs().max()) for k in qs)
    print(f"{a.full}: {len(qs)} linear matrices quantised, max abs error {err:.4f}", flush=True)
    write(a.full, a.out, {k: dequantise(*qs[k]) if k in qs else v for k, v in full.items()})
    for src, out in zip(a.slice[::2], a.slice[1::2]):
        n = load(src)
        bad = [k for k, v in n.items() if is_linear(k) and not torch.equal(v, full[k][:v.shape[0], :v.shape[1]])]
        if bad:
            raise SystemExit(f"{src}: {len(bad)} linear weights are not prefixes of {a.full}, e.g. {bad[:2]}")
        new = {}
        for k, v in n.items():
            if is_linear(k):
                q, s = qs[k]
                new[k] = dequantise(q[:v.shape[0], :v.shape[1]], s[:v.shape[0]])
            else:
                new[k] = v
        write(src, out, new)
        print(f"{src} -> {out} (full model's int8 weights, sliced)", flush=True)


if __name__ == "__main__":
    main()
