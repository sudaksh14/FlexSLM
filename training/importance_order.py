"""
M1 of the MedNest recipe: importance ordering of a Hugging Face Qwen3/LLaMA
checkpoint before nested training.

FlexLLaMA levels are prefixes: level k keeps the first hidden_dims[k] residual
channels, the first num_kv_heads[k] KV groups (with their query heads) and the
first intermediate_dims[k] MLP channels. Ordering permutes the checkpoint once so
that those prefixes hold the most important structures; the permuted model
computes the same function at full width (checked here), only the sub-levels
change.

Scores (first-order Taylor, mean over calibration tokens of |x * dL/dx|, with L
the next-token cross-entropy of the full model; ported from Elastoformer-MOO
importance.py, method "taylor"):
  * MLP channel j: x = input of down_proj (activated gate * up), per channel.
  * KV group g:    x = input of o_proj, summed over head_dim per query head, then
                   over the query heads of group g.
  * residual channel c (Elastoformer does not nest the hidden size; MedNest does):
                   x = residual stream at the input of every RMSNorm (both per
                   layer and the final one), summed over those sites. This is a
                   heuristic: zeroing c removes it from all sites at once, which
                   a sum of per-site first-order terms only approximates. It
                   follows Minitron's activation-based embedding-channel score,
                   with the Taylor term in place of the activation.
Gradients are taken w.r.t. activations only (a leaf embedding input), never
w.r.t. parameters.

Permutation (new position -> old index, most important first, stable ties):
  * MLP: rows of gate/up, columns of down.
  * KV group: rows of q (the group's query heads), k, v; columns of o. q_norm and
    k_norm act on head_dim and are untouched.
  * residual: columns of embed_tokens (and of an untied lm_head), of q/k/v/gate/up;
    rows of o/down; the weights of every RMSNorm on the residual stream. RMSNorm's
    mean of squares over the full width is permutation-invariant, so the full model
    is unchanged.

Calibration: windows from the PMC train split (data/pmc_qwen3/train.bin, the
Stage-1 training data; never val/test), deterministic offsets.

GPU job on DAS-6 (eval GPUs):
  sbatch scripts/das6/importance_order.sbatch
  python -m training.importance_order --model Qwen/Qwen3-0.6B --out hf_export/qwen3-0.6b.taylor
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# Ordering schemes (Stage-1 pilot dimension, 2026-11-10: untrained Taylor ordering made the
# sub-levels worse, see implementation_log.md, so the pilots decide):
#   "none"       the original Qwen3-0.6B weight order (Hugging Face checkpoint as is)
#   "hidden_act" residual channels by mean |x| at the RMSNorm inputs (Minitron-style),
#                MLP channels and KV groups in the original order
#   "taylor"     everything by the Taylor score below
#   "minitron"   everything by activation magnitude, exactly as NVIDIA Model Optimizer's
#                mcore_minitron (Minitron, Flextron, LLaMaFlex), see collect_minitron_importance
ORDERINGS = ("none", "hidden_act", "taylor", "minitron")
ORDERED_CHECKPOINTS = {"taylor": os.path.join("hf_export", "qwen3-0.6b.taylor"),
                       "hidden_act": os.path.join("hf_export", "qwen3-0.6b.hidden_act"),
                       "minitron": os.path.join("hf_export", "qwen3-0.6b.minitron")}
ORDERED_QWEN3_0P6B = ORDERED_CHECKPOINTS["taylor"]
# Ordering of the final recipe (s1.qwen3_0p6b.full and the Stage-2 runs that need a start
# checkpoint). Provisional: set from the pilots s1.qwen3_0p6b.pilot.{base,order_*}.
FINAL_ORDERING = "none"


def start_weights(ordering: str, hf_model: str = "Qwen/Qwen3-0.6B") -> str:
    """Warm-start checkpoint for an ordering scheme of Qwen3-0.6B."""
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    return hf_model if ordering == "none" else ORDERED_CHECKPOINTS[ordering]


def _layers(model):
    return model.model.layers


def calibration_batches(data_dir: str, n_seq: int, seq_len: int, batch_size: int, seed: int = 4242,
                        split: str = "train"):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=meta["dtype"], mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - seq_len - 1, size=n_seq)
    for lo in range(0, n_seq, batch_size):
        rows = [np.asarray(data[s:s + seq_len], dtype=np.int64) for s in starts[lo:lo + batch_size]]
        yield torch.from_numpy(np.stack(rows))


def collect_taylor_importance(model, batches, skip_first_token: bool = False) -> dict:
    """Per layer: 'mlp' [intermediate], 'groups' [num_kv_heads]; 'hidden' [hidden_size]."""
    cfg = model.config
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    first = 1 if skip_first_token else 0
    layers = _layers(model)
    device = next(model.parameters()).device
    mlp_sum = [torch.zeros(cfg.intermediate_size, device=device) for _ in layers]
    head_sum = [torch.zeros(n_heads, device=device) for _ in layers]
    hidden_sum = torch.zeros(cfg.hidden_size, device=device)
    tokens = 0

    def on_grad(x, fn):
        # The hook captures a detached view, not x itself: x -> hook -> x would be a
        # reference cycle that keeps every batch's autograd graph alive (jobs 28091/28094
        # ran out of memory on a 24 GB A10 that way).
        xd = x.detach()

        def hook(g):
            fn((xd * g).float())  # returns None: the gradient itself is left unchanged
        x.register_hook(hook)

    handles = []
    for idx, layer in enumerate(layers):
        def mlp_hook(_m, args, idx=idx):
            on_grad(args[0], lambda p: mlp_sum[idx].add_(p[:, first:].abs().sum(dim=(0, 1))))

        def attn_hook(_m, args, idx=idx):
            x = args[0]
            b, t, _ = x.shape
            on_grad(x, lambda p: head_sum[idx].add_(
                p.view(b, t, n_heads, head_dim).sum(-1)[:, first:].abs().sum(dim=(0, 1))))

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_hook))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(attn_hook))

    def resid_hook(_m, args):
        on_grad(args[0], lambda p: hidden_sum.add_(p[:, first:].abs().sum(dim=(0, 1))))

    for norm in _residual_norms(model):
        handles.append(norm.register_forward_pre_hook(resid_hook))

    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            batch = batch.to(device)
            emb = model.get_input_embeddings()(batch).detach().requires_grad_(True)
            logits = model(inputs_embeds=emb, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), batch[:, 1:].flatten())
            loss.backward(inputs=[emb])
            tokens += batch.shape[0] * (batch.shape[1] - first)
            del emb, logits, loss
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    if tokens == 0:
        raise ValueError("no calibration tokens were processed")
    group = n_heads // n_kv
    return dict(
        mlp=[(s / tokens).cpu() for s in mlp_sum],
        groups=[(s / tokens).view(n_kv, group).sum(1).cpu() for s in head_sum],
        hidden=(hidden_sum / tokens).cpu(),
        tokens=tokens,
    )


@torch.no_grad()
def collect_hidden_activation(model, batches) -> torch.Tensor:
    """Residual channel score: mean over tokens of |x_c| at the input of every RMSNorm on
    the residual stream, summed over those sites (Minitron's activation criterion)."""
    device = next(model.parameters()).device
    total = torch.zeros(model.config.hidden_size, device=device)
    tokens = 0

    def hook(_m, args):  # returns None: a pre-hook's return value would replace the input
        total.add_(args[0].float().abs().sum(dim=(0, 1)))

    handles = [n.register_forward_pre_hook(hook) for n in _residual_norms(model)]
    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            model(input_ids=batch.to(device), use_cache=False)
            tokens += batch.numel()
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return (total / max(1, tokens)).cpu()


@torch.no_grad()
def collect_minitron_importance(model, batches) -> dict:
    """Activation-magnitude scores, forward passes only, ported from NVIDIA Model Optimizer
    (modelopt/torch/prune/plugins/mcore_minitron.py; Muralidharan et al. 2024, used by
    Flextron and LLaMaFlex). Aggregation everywhere: mean of |x| over the sequence, then
    L2 over the calibration sequences (Minitron's best (batch L2, sequence mean)).
      * MLP channel: input of down_proj.
      * head: per channel of the o_proj input, then L2 over the head's head_dim channels.
        Minitron keeps the KV groups and ranks query heads inside each group; MedNest
        levels drop whole KV groups, so a group scores the L2 over all its channels
        (modelopt's commented-out group_importance).
      * residual channel: OUTPUT of input_layernorm and post_attention_layernorm
        (normalised and scaled, unlike hidden_act, which reads the raw norm inputs),
        L2 per site, summed over sites; the final norm is not used (as in modelopt)."""
    cfg = model.config
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    layers = _layers(model)
    device = next(model.parameters()).device

    def acc(buf, x):  # [b, t, c] -> buf [c] += sum_b (mean_t |x|)^2; returns None (keeps outputs)
        buf.add_(x.float().abs().mean(dim=1).pow(2).sum(dim=0))
    mlp = [torch.zeros(cfg.intermediate_size, device=device) for _ in layers]
    attn = [torch.zeros(n_heads * head_dim, device=device) for _ in layers]
    hidden = [torch.zeros(cfg.hidden_size, device=device) for _ in range(2 * len(layers))]
    handles = []
    for idx, layer in enumerate(layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda _m, args, idx=idx: acc(mlp[idx], args[0])))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda _m, args, idx=idx: acc(attn[idx], args[0])))
        for k, norm in enumerate((layer.input_layernorm, layer.post_attention_layernorm)):
            handles.append(norm.register_forward_hook(
                lambda _m, _a, out, site=2 * idx + k: acc(hidden[site], out)))
    was_training = model.training
    model.eval()
    tokens = 0
    try:
        for batch in batches:
            model(input_ids=batch.to(device), use_cache=False)
            tokens += batch.numel()
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    if tokens == 0:
        raise ValueError("no calibration tokens were processed")
    group = n_heads // n_kv
    return dict(
        mlp=[s.sqrt().cpu() for s in mlp],
        groups=[s.view(n_kv, group * head_dim).sum(1).sqrt().cpu() for s in attn],
        hidden=torch.stack([s.sqrt() for s in hidden]).sum(0).cpu(),
        tokens=tokens,
    )


def _residual_norms(model):
    norms = []
    for layer in _layers(model):
        norms += [layer.input_layernorm, layer.post_attention_layernorm]
    return norms + [model.model.norm]


def permutation_from_importance(imp: dict) -> dict:
    order = lambda s: torch.argsort(s, descending=True, stable=True)
    return dict(mlp=[order(s) for s in imp["mlp"]], groups=[order(s) for s in imp["groups"]],
                hidden=order(imp["hidden"]))


def _block_index(order: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.cat([torch.arange(g * rows, (g + 1) * rows) for g in order.tolist()])


@torch.no_grad()
def apply_permutation(model, perm: dict) -> None:
    """Permute a Hugging Face llama/qwen2/qwen3 causal LM in place."""
    cfg = model.config
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    group = n_heads // n_kv
    h = perm["hidden"]

    def take(p, index, dim):
        p.data = p.data.index_select(dim, index.to(p.device)).clone()

    emb = model.get_input_embeddings()
    take(emb.weight, h, 1)
    lm_head = model.get_output_embeddings()
    if lm_head is not None and lm_head.weight is not emb.weight:
        take(lm_head.weight, h, 1)
    for norm in _residual_norms(model):
        take(norm.weight, h, 0)

    for layer, mlp_order, group_order in zip(_layers(model), perm["mlp"], perm["groups"]):
        mlp, attn = layer.mlp, layer.self_attn
        for proj in (mlp.gate_proj, mlp.up_proj):
            take(proj.weight, mlp_order, 0)
            take(proj.weight, h, 1)
            if proj.bias is not None:
                take(proj.bias, mlp_order, 0)
        take(mlp.down_proj.weight, mlp_order, 1)
        take(mlp.down_proj.weight, h, 0)

        q_idx = _block_index(group_order, group * head_dim)
        kv_idx = _block_index(group_order, head_dim)
        for proj, index in ((attn.q_proj, q_idx), (attn.k_proj, kv_idx), (attn.v_proj, kv_idx)):
            take(proj.weight, index, 0)
            take(proj.weight, h, 1)
            if proj.bias is not None:
                take(proj.bias, index, 0)
        take(attn.o_proj.weight, q_idx, 1)
        take(attn.o_proj.weight, h, 0)
        if getattr(attn.o_proj, "bias", None) is not None:
            take(attn.o_proj.bias, h, 0)


@torch.no_grad()
def permuted_logit_diff(model, perm: dict, batch: torch.Tensor) -> tuple[float, float]:
    """Apply perm in place; return (max |logit diff|, max |logit|) on batch."""
    model.eval()
    device = next(model.parameters()).device
    before = model(input_ids=batch.to(device)).logits.float()
    apply_permutation(model, perm)
    after = model(input_ids=batch.to(device)).logits.float()
    return (before - after).abs().max().item(), before.abs().max().item()


@torch.no_grad()
def flex_level_ppl(hf_path: str, batches, fractions=(0.5, 0.75, 1.0), preset: str = "qwen3-0.6b") -> list:
    """Per-level perplexity of the FlexLLaMA warm-started from hf_path (no training)."""
    import dataclasses
    import utils
    from config.experiments import make_flexllama_warmstart_kd
    cfg = make_flexllama_warmstart_kd(preset, fractions=fractions).model_config
    cfg = dataclasses.replace(cfg, pretrained_hf_model=hf_path)
    model = cfg.make_model()
    utils.load_llama_weights_into_flexllama(model, hf_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    batches = list(batches)  # reused for every level
    out = []
    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        nll, n = 0.0, 0
        for batch in batches:
            batch = batch.to(device)
            logits = model(batch)
            nll += F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), batch[:, 1:].flatten(),
                                   reduction="sum").item()
            n += batch[:, 1:].numel()
        out.append(float(np.exp(nll / n)))
    del model
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--scheme", choices=[o for o in ORDERINGS if o != "none"], default="taylor")
    p.add_argument("--out", default=None, help="default: ORDERED_CHECKPOINTS[scheme]")
    p.add_argument("--data-dir", default=os.path.join("data", "pmc_qwen3"))
    p.add_argument("--calib-seqs", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)  # 4 x 1,024 fp32 ran out of memory on a 24 GB A10 (job 28088)
    p.add_argument("--skip-first-token", action="store_true")
    p.add_argument("--eval-seqs", type=int, default=64, help="PMC val windows for the per-level ppl check (0: skip)")
    args = p.parse_args()
    args.out = args.out or ORDERED_CHECKPOINTS[args.scheme]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    calib = calibration_batches(args.data_dir, args.calib_seqs, args.seq_len, args.batch_size)
    # Scoring cost (the only part that differs between schemes; permutation, check and
    # ppl evaluation are the same for all): wall time and peak GPU memory.
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_score = time.time()
    if args.scheme in ("taylor", "minitron"):
        imp = (collect_taylor_importance(model, calib, skip_first_token=args.skip_first_token)
               if args.scheme == "taylor" else collect_minitron_importance(model, calib))
        perm = permutation_from_importance(imp)
    else:  # hidden_act: only the residual channels move
        hidden = collect_hidden_activation(model, calib)
        imp = dict(hidden=hidden, tokens=args.calib_seqs * args.seq_len)
        n_layers = model.config.num_hidden_layers
        perm = dict(mlp=[torch.arange(model.config.intermediate_size)] * n_layers,
                    groups=[torch.arange(model.config.num_key_value_heads)] * n_layers,
                    hidden=torch.argsort(hidden, descending=True, stable=True))
    if device == "cuda":
        torch.cuda.synchronize()
    score_seconds = time.time() - t_score
    score_peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
    print(f"[score] scheme {args.scheme}: {score_seconds:.1f} s, peak GPU memory {score_peak_gb:.2f} GB "
          f"({torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}, batch {args.batch_size})", flush=True)
    check = next(calibration_batches(args.data_dir, 2, 256, 2, seed=99, split="val"))
    diff, scale = permuted_logit_diff(model, perm, check)
    tol = 1e-4 + 1e-4 * scale
    print(f"importance on {imp['tokens']:,} tokens; permuted full model max |logit diff| {diff:.2e} "
          f"(max |logit| {scale:.1f}, tolerance {tol:.2e})", flush=True)
    if diff > tol:
        raise SystemExit("permutation changed the full model: not saving")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    torch.save(dict(importance=imp, permutation=perm), os.path.join(args.out, "importance.pt"))
    meta = dict(source=args.model, method={"taylor": "taylor |x*dL/dx|", "hidden_act": "hidden mean |x| at RMSNorm inputs",
                             "minitron": "activation magnitude (modelopt mcore_minitron)"}[args.scheme],
                calib=f"{args.data_dir}/train.bin", score_seconds=round(score_seconds, 1),
                score_peak_gb=round(score_peak_gb, 2), batch_size=args.batch_size,
                calib_seqs=args.calib_seqs, seq_len=args.seq_len, skip_first_token=args.skip_first_token,
                tokens=imp["tokens"], max_logit_diff=diff)
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    if args.eval_seqs:
        ev = lambda: calibration_batches(args.data_dir, args.eval_seqs, args.seq_len, args.batch_size,
                                         seed=7, split="val")
        meta["val_ppl_unordered"] = flex_level_ppl(args.model, ev())
        meta["val_ppl_ordered"] = flex_level_ppl(args.out, ev())
        print(f"PMC val ppl per level (0.5 / 0.75 / 1.0), no training: "
              f"unordered {meta['val_ppl_unordered']}  ordered {meta['val_ppl_ordered']}", flush=True)
    meta["seconds"] = round(time.time() - t0)
    with open(os.path.join(args.out, "ordering.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
