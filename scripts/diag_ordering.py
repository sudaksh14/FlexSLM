"""
Diagnostic for M1 (job 28100 found Taylor ordering *raised* untrained level ppl:
L0 4,265 -> 26,698, L1 313 -> 605). Applies each part of the ordering on its own and
prints the untrained per-level PMC val ppl of every variant:
  none | mlp | groups | hidden | mlp+groups | all (taylor) | all with an
  activation-magnitude hidden score (Minitron-style mean |x| at the RMSNorm inputs).
    python scripts/diag_ordering.py   (GPU, after training.importance_order)
"""
import copy
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.getcwd())
from training.importance_order import (ORDERED_QWEN3_0P6B, _residual_norms, apply_permutation,
                                       calibration_batches, flex_level_ppl)

DATA = os.path.join("data", "pmc_qwen3")


@torch.no_grad()
def activation_hidden_score(model, batches):
    total = torch.zeros(model.config.hidden_size, device=next(model.parameters()).device)
    def hook(_m, args):  # returns None: a pre-hook's return value would replace the input
        total.add_(args[0].float().abs().sum(dim=(0, 1)))
    hooks = [n.register_forward_pre_hook(hook) for n in _residual_norms(model)]
    for b in batches:
        model(input_ids=b.to(total.device), use_cache=False)
    for h in hooks:
        h.remove()
    return total


@torch.no_grad()
def act_weight_scores(model, batches):
    """Elastoformer-MOO's default ('act_weight'): MLP channel j = mean |h_j| * ||down[:, j]||,
    KV group = sum over its query heads of mean ||head output|| * ||o_proj columns||."""
    cfg = model.config
    H, G = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    dev = next(model.parameters()).device
    layers = model.model.layers
    mlp = [torch.zeros(cfg.intermediate_size, device=dev) for _ in layers]
    heads = [torch.zeros(H, device=dev) for _ in layers]
    hooks = []
    for i, layer in enumerate(layers):
        def mh(_m, a, i=i):
            mlp[i].add_(a[0].float().abs().sum(dim=(0, 1)))
        def ah(_m, a, i=i):
            x = a[0].float()
            heads[i].add_(x.view(*x.shape[:2], H, hd).norm(dim=-1).sum(dim=(0, 1)))
        hooks += [layer.mlp.down_proj.register_forward_pre_hook(mh),
                  layer.self_attn.o_proj.register_forward_pre_hook(ah)]
    for b in batches:
        model(input_ids=b.to(dev), use_cache=False)
    for h in hooks:
        h.remove()
    order = lambda v: torch.argsort(v.cpu(), descending=True, stable=True)
    m_out, g_out = [], []
    for i, layer in enumerate(layers):
        m_out.append(order(mlp[i] * layer.mlp.down_proj.weight.float().norm(dim=0)))
        o = layer.self_attn.o_proj.weight.float()
        hs = heads[i] * o.view(o.shape[0], H, hd).norm(dim=(0, 2))
        g_out.append(order(hs.view(G, H // G).sum(1)))
    return m_out, g_out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    blob = torch.load(os.path.join(ORDERED_QWEN3_0P6B, "importance.pt"), weights_only=False)
    perm = blob["permutation"]
    n_layers = len(perm["mlp"])
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32).cuda().eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ident = dict(mlp=[torch.arange(len(p)) for p in perm["mlp"]],
                 groups=[torch.arange(len(p)) for p in perm["groups"]],
                 hidden=torch.arange(len(perm["hidden"])))
    act = activation_hidden_score(base, calibration_batches(DATA, 128, 512, 4))
    act_order = torch.argsort(act.cpu(), descending=True, stable=True)
    print("hidden: overlap of first 512 channels, taylor vs activation:",
          len(set(perm["hidden"][:512].tolist()) & set(act_order[:512].tolist())),
          "| taylor vs original:", int((perm["hidden"][:512] < 512).sum()),
          "| activation vs original:", int((act_order[:512] < 512).sum()), flush=True)
    top = torch.argsort(act.cpu(), descending=True)[:8]
    print("largest-activation channels:", top.tolist(), "taylor rank:",
          [int((perm["hidden"] == c).nonzero()) for c in top], flush=True)

    aw_mlp, aw_groups = act_weight_scores(base, calibration_batches(DATA, 128, 512, 4))
    rev = lambda ps: [p.flip(0) for p in ps]
    variants = {
        "mlp_actweight": dict(ident, mlp=aw_mlp),
        "groups_actweight": dict(ident, groups=aw_groups),
        "mlp_taylor_reversed": dict(ident, mlp=rev(perm["mlp"])),
        "groups_taylor_reversed": dict(ident, groups=rev(perm["groups"])),
        "actweight+hidden_act": dict(mlp=aw_mlp, groups=aw_groups, hidden=act_order),
    }
    variants_first_run = {
        "none": ident,
        "mlp": dict(ident, mlp=perm["mlp"]),
        "groups": dict(ident, groups=perm["groups"]),
        "hidden_taylor": dict(ident, hidden=perm["hidden"]),
        "mlp+groups": dict(ident, mlp=perm["mlp"], groups=perm["groups"]),
        "all_taylor": perm,
        "mlp+groups+hidden_act": dict(perm, hidden=act_order),
        "hidden_act": dict(ident, hidden=act_order),
    }
    ev = list(calibration_batches(DATA, 32, 512, 4, seed=7, split="val"))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        for name, p in variants.items():
            m = copy.deepcopy(base)
            apply_permutation(m, p)
            path = os.path.join(d, name)
            m.save_pretrained(path)
            tok.save_pretrained(path)
            del m
            print(f"{name:24s} ppl L0/L1/L2 = " + " / ".join(f"{x:,.1f}" for x in flex_level_ppl(path, ev)),
                  flush=True)
    assert n_layers == base.config.num_hidden_layers


if __name__ == "__main__":
    main()
