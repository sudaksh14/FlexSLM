"""
LoRA fine-tuning of a large Hugging Face model as the Stage-2 KD teacher (answer-only medmix SFT).

Why this exists: Stage 2 distils from a teacher that answers in the answer-only format
("\\boxed{X}<|im_end|>" after an empty think block). The fine-tuned Qwen3-1.7B (E_A) is that teacher today;
this script builds a larger one (default: HuatuoGPT-o1-7B, Qwen2.5-7B based, PMC ppl 5.01 vs 5.57 for the
fine-tuned 1.7B) with LoRA, so it fits on hipster's 48 GB cards without optimizer state for 7B weights.

What it does, and the choices that make the result a drop-in `teacher_hf_model` for training/sft.py:
  - data: data/medmix_sft/{train,val}.jsonl through training.sft.QADataset / collate_qa, tokenized with the
    STUDENT's tokenizer (Qwen3), so the teacher sees exactly the token ids the student sees (KD needs identical ids);
    loss = CE on the answer tokens only, as in the Stage-2 teacher run E_A;
  - the LM head is applied to the answer positions only (a handful of tokens per row), so a 152k vocabulary never
    produces [batch, seq, vocab] logits;
  - LoRA on all attention + MLP projections, bf16 base, activation checkpointing, DDP (torchrun) one process per GPU;
  - no gradient clipping by default (the repo's Stage-2 finding: clipping hurt the 1.7B teacher); grad_norm is logged;
  - at the end the adapter is merged and the vocabulary is trimmed to the student's 151,936 rows (Qwen2.5 pads the
    embedding matrix to 152,064; the extra ids never occur, and training/trainer.py softmaxes the teacher logits as
    [.., student_vocab]), then saved as an HF directory next to the Qwen3 tokenizer.

    torchrun --standalone --nproc_per_node=4 -m training.finetune_teacher_lora            # full run
    SMOKE: add --smoke_steps 20 (longest batches first, no merge unless --smoke_merge_dir is given)
Launched on hipster by scripts/hipster/teacher_lora.sbatch.
"""
import argparse
import datetime
import json
import math
import os
import random
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100
STUDENT_VOCAB = 151_936            # Qwen3 (0.6B / 1.7B) embedding rows; the teacher's logits must match
THINK_IDS = (151_667, 151_668)     # <think>, </think>: rows of Qwen2.5's padded embedding matrix (untrained)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="FreedomIntelligence/HuatuoGPT-o1-7B")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="the student's tokenizer (same ids as the Qwen3 family)")
    p.add_argument("--data_dir", default=os.path.join("data", "medmix_sft"))
    p.add_argument("--out", default=os.path.join("hf_export", "medmix_huatuogpt_o1_7b_lora"),
                   help="merged HF directory (the future teacher_hf_model)")
    p.add_argument("--adapter_dir", default=None, help="adapter checkpoints (default $FLEXSLM_STORE/teacher_lora/<job>)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--global_batch", type=int, default=32, help="sequences per optimizer step (FaithMed schedule)")
    p.add_argument("--micro_batch", type=int, default=4, help="sequences per GPU per forward")
    p.add_argument("--max_seq", type=int, default=3072)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.03, help="fraction of the steps")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0, help="0 = no clipping (grad_norm still logged)")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_steps", type=int, default=0, help="N > 0: N steps on the longest sequences, then stop")
    p.add_argument("--smoke_merge_dir", default=None, help="smoke only: also merge + save + verify here (deleted after)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------- model pieces
def wrap_lora(model, rank, alpha, dropout, targets=LORA_TARGETS):
    """Freeze `model` and add LoRA adapters (peft) on every attention and MLP projection."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets),
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


class AnswerLM(nn.Module):
    """forward(input_ids, labels) -> (sum of CE over the answer tokens, number of answer tokens, correct argmax).
    Runs the transformer body, then the LM head on the answer positions only. Wrapped by DDP as a whole, so the
    gradient all-reduce hooks fire (calling a sub-module of a DDP model directly would bypass them).
    Right padding + causal attention: real tokens never see the padding, so no attention mask is needed."""

    def __init__(self, peft_model):
        super().__init__()
        self.peft = peft_model

    def forward(self, input_ids, labels):
        inner = self.peft.base_model.model              # the HF *ForCausalLM with LoRA layers injected
        hidden = inner.model(input_ids=input_ids).last_hidden_state[:, :-1]
        targets = labels[:, 1:]
        sel = targets != IGNORE_INDEX
        logits = inner.lm_head(hidden[sel]).float()
        t = targets[sel]
        ce = F.cross_entropy(logits, t, reduction="sum")
        return ce, t.numel(), (logits.argmax(-1) == t).sum()


def merge_and_trim(peft_model, vocab_size=STUDENT_VOCAB):
    """Merge the adapter into the base weights and cut the (padded) vocabulary down to the student's rows."""
    merged = peft_model.merge_and_unload()
    if merged.get_input_embeddings().weight.shape[0] != vocab_size:
        merged.resize_token_embeddings(vocab_size)
    merged.config.vocab_size = vocab_size
    merged.config.use_cache = True
    if hasattr(merged, "gradient_checkpointing_disable"):
        merged.gradient_checkpointing_disable()
    return merged


def check_export(path, vocab_size=STUDENT_VOCAB):
    """Reads the saved safetensors headers: embedding and LM head have exactly `vocab_size` rows."""
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(path, "config.json")))
    assert cfg["vocab_size"] == vocab_size, cfg["vocab_size"]
    index = os.path.join(path, "model.safetensors.index.json")
    wmap = json.load(open(index))["weight_map"] if os.path.exists(index) else None
    shapes = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if name == "lm_head.weight" and cfg.get("tie_word_embeddings"):
            continue                                    # tied: stored once, as the embedding
        shard = os.path.join(path, wmap[name] if wmap else "model.safetensors")
        with safe_open(shard, framework="pt") as f:
            shapes[name] = tuple(f.get_slice(name).get_shape())
        assert shapes[name][0] == vocab_size, (name, shapes[name])
    return cfg["model_type"], shapes


# ------------------------------------------------------------------------------------------------- data / loops
def lr_at(step, total, warmup, lr):
    w = max(1, int(round(warmup * total)))
    if step < w:
        return lr * (step + 1) / w
    return 0.5 * lr * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def step_plan(n_items, global_batch, epochs, seed, lengths=None, longest_first=False):
    """List of steps, each a list of `global_batch` dataset indices (drop_last, reshuffled per epoch)."""
    per_epoch = n_items // global_batch
    steps = []
    for e in range(math.ceil(epochs)):
        order = list(range(n_items))
        random.Random(seed + e).shuffle(order)
        if longest_first:
            order.sort(key=lambda i: -lengths[i])
        for s in range(per_epoch):
            steps.append(order[s * global_batch:(s + 1) * global_batch])
    return steps[:int(round(per_epoch * epochs))]


def run_batches(ddp_model, items, collate, indices, micro, device, backward_scale=None, no_sync_last=True):
    """Forward (and, with backward_scale, backward) of one rank's share, in micro-batches.
    Returns (ce_sum, answer_tokens, correct, real_tokens)."""
    ce_tot = torch.zeros((), device=device)
    corr_tot = torch.zeros((), device=device)
    n_tot = 0
    real = 0
    chunks = [indices[i:i + micro] for i in range(0, len(indices), micro)]
    for ci, chunk in enumerate(chunks):
        input_ids, labels, mask = collate([items[i] for i in chunk])
        input_ids, labels = input_ids.to(device), labels.to(device)
        real += int(mask.sum())
        last = ci == len(chunks) - 1
        ctx = ddp_model.no_sync() if (backward_scale is not None and not last and hasattr(ddp_model, "no_sync")) \
            else _null()
        with ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                ce, n, corr = ddp_model(input_ids, labels)
            if backward_scale is not None:
                (ce * backward_scale).backward()
        ce_tot += ce.detach()
        corr_tot += corr.detach()
        n_tot += int(n)
    return ce_tot, n_tot, corr_tot, real


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def evaluate(ddp_model, items, collate, micro, device, rank, world):
    mine = list(range(rank, len(items), world))
    ddp_model.eval()
    with torch.no_grad():
        ce, n, corr, _ = run_batches(ddp_model, items, collate, mine, micro, device)
    ddp_model.train()
    t = torch.tensor([float(ce), float(n), float(corr)], device=device)
    if world > 1:
        dist.all_reduce(t)
    return t[0].item() / max(1.0, t[1].item()), t[2].item() / max(1.0, t[1].item())


def main(argv=None):
    a = parse_args(argv)
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", local) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(hours=3))
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    assert a.global_batch % world == 0, "global_batch must be divisible by the number of GPUs"
    per_rank = a.global_batch // world
    random.seed(a.seed + rank)
    torch.manual_seed(a.seed)

    # Imported here: pulls in the repo's training stack (only needed for the data classes).
    from training.sft import QADataset, collate_qa
    from transformers import AutoModelForCausalLM
    train =QADataset(os.path.join(a.data_dir, "train.jsonl"), a.tokenizer, a.max_seq)
    val = QADataset(os.path.join(a.data_dir, "val.jsonl"), a.tokenizer, a.max_seq)
    lengths = [len(p) + len(x) for p, x in train.items]
    log(f"[data] train {len(train)} val {len(val)} truncated {train.num_truncated} "
        f"mean len {sum(lengths) / len(lengths):.0f} max {max(lengths)}  real tokens/epoch {sum(lengths):,}")

    t0 = time.time()
    # device_map loads each rank's copy straight onto its GPU (no 15 GB host copy per rank).
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16 if cuda else torch.float32,
                                                attn_implementation="sdpa",
                                                **({"device_map": {"": local}} if cuda else {}))
    base.config.use_cache = False
    emb = base.get_input_embeddings().weight
    log(f"[model] {a.base}: {sum(q.numel() for q in base.parameters()) / 1e9:.2f}B params, embedding rows {emb.shape[0]}, "
        f"loaded in {time.time() - t0:.0f} s")
    if emb.shape[0] > max(THINK_IDS):
        norms = emb.float().norm(dim=-1)
        log(f"[model] embedding row norms: median {norms.median():.3f}; <|im_start|>/<|im_end|> "
            f"{norms[151644]:.3f}/{norms[151645]:.3f}; <think>/</think> (untrained rows in Qwen2.5-based models) "
            f"{norms[THINK_IDS[0]]:.3f}/{norms[THINK_IDS[1]]:.3f}")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = wrap_lora(base, a.rank, a.alpha, a.dropout)
    peft_model.to(device)
    if rank == 0:
        peft_model.print_trainable_parameters()
    model = AnswerLM(peft_model).to(device)
    # broadcast_buffers=False: no collective in forward, so ranks may run different numbers of eval batches.
    ddp = (nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None, broadcast_buffers=False)
           if world > 1 else model)
    params = [q for q in peft_model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=a.weight_decay)

    smoke = a.smoke_steps > 0
    plan = step_plan(len(train), a.global_batch, a.epochs, a.seed, lengths, longest_first=smoke)
    if smoke:
        plan = plan[:a.smoke_steps]
    total = len(plan)
    job = os.environ.get("SLURM_JOB_ID", "local")
    adapter_dir = a.adapter_dir or os.path.join(os.environ.get("FLEXSLM_STORE", "store"), "teacher_lora", job)
    log(f"[train] {total} steps, global batch {a.global_batch} = {world} GPUs x {per_rank} (micro {a.micro_batch}), "
        f"lr {a.lr} warmup {a.warmup} cosine, LoRA r {a.rank} alpha {a.alpha}, clip {a.grad_clip or 'off'}, smoke={smoke}")

    ddp.train()
    start = time.time()
    tok_window, t_window = 0, time.time()
    step_times, tok_per_s = [], []
    for step, idx in enumerate(plan):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, a.warmup, a.lr)
        mine = idx[rank::world]
        n_local = sum(len(train.items[i][1]) for i in mine)           # answer tokens of this rank's share
        ts = time.time()
        ce, n, corr, real = run_batches(ddp, train.items, collate_qa, mine, a.micro_batch, device,
                                        backward_scale=1.0 / max(1, n_local))
        gn = torch.nn.utils.clip_grad_norm_(params, a.grad_clip if a.grad_clip > 0 else float("inf"))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - ts
        step_times.append(dt)
        tok_per_s.append(real / dt)
        tok_window += real
        if (step + 1) % 10 == 0 or step == 0 or step + 1 == total:
            el = time.time() - start
            log(f"[step {step + 1}/{total}] loss {ce.item() / max(1, n):.4f} acc {corr.item() / max(1, n):.3f} "
                f"grad_norm {float(gn):.3f} lr {opt.param_groups[0]['lr']:.2e} "
                f"{tok_window / (time.time() - t_window):.0f} tok/s/rank step {dt:.1f}s "
                f"max_mem {torch.cuda.max_memory_allocated() / 2**30 if cuda else 0:.1f}GB elapsed {el / 60:.1f} min "
                f"eta {(total - step - 1) * el / (step + 1) / 60:.0f} min")
            tok_window, t_window = 0, time.time()
        if not smoke and (step + 1) % a.eval_every == 0:
            vl, va = evaluate(ddp, val.items, collate_qa, a.micro_batch * 2, device, rank, world)
            log(f"[val step {step + 1}] loss {vl:.4f} answer-token acc {va:.3f}")
        if not smoke and (step + 1) % a.save_every == 0 and rank == 0:
            peft_model.save_pretrained(os.path.join(adapter_dir, f"step{step + 1}"))

    vl, va = evaluate(ddp, val.items, collate_qa, a.micro_batch * 2, device, rank, world)
    log(f"[val final] loss {vl:.4f} answer-token acc {va:.3f}")

    if smoke:
        mean_t = sum(step_times[1:]) / max(1, len(step_times) - 1)
        full_steps = len(step_plan(len(train), a.global_batch, a.epochs, a.seed))
        real_tok = sum(lengths) * a.epochs
        mean_tok = sum(tok_per_s[1:]) / max(1, len(tok_per_s) - 1)
        log(f"[smoke] mean step {mean_t:.1f} s on the LONGEST batches ({mean_tok:.0f} tok/s/rank); "
            f"pessimistic full run = {full_steps} steps x {mean_t:.1f} s = {full_steps * mean_t / 3600:.1f} h; "
            f"token-based = {real_tok / (mean_tok * world) / 3600:.1f} h (real tokens, longest-batch efficiency); "
            f"peak memory {torch.cuda.max_memory_allocated() / 2**30 if cuda else 0:.1f} GB")
        if a.smoke_merge_dir and rank == 0:
            _merge_save(peft_model, a, a.smoke_merge_dir, log)
            shutil.rmtree(a.smoke_merge_dir, ignore_errors=True)
            log(f"[smoke] merge/save/verify ok, {a.smoke_merge_dir} removed")
    else:
        if rank == 0:
            peft_model.save_pretrained(os.path.join(adapter_dir, "final"))
            _merge_save(peft_model, a, a.out, log)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    log("[teacher_lora] done")


def _merge_save(peft_model, a, out, log):
    from transformers import AutoTokenizer
    t0 = time.time()
    merged = merge_and_trim(peft_model)
    os.makedirs(out, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(a.tokenizer).save_pretrained(out)
    kind, shapes = check_export(out)
    log(f"[export] {out} ({kind}) vocab {STUDENT_VOCAB}: {shapes}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
