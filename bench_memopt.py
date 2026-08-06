"""
A/B peak-GPU-memory benchmark: training.py trainer vs training_memopt.py trainer,
on the real Qwen2.5-1.5B warm-start config dims (V=151936). Single GPU - DDP
replicates per-rank, so per-GPU peak is what decides whether a config fits.

Run via bench_memopt.sh (sbatch), not directly.
"""
import sys
import types
import torch

from config.experiments import CONFIGS


def instrument(module):
    module.log = lambda *a, **k: None
    module.optimizers = lambda: types.SimpleNamespace(
        zero_grad=lambda: module.zero_grad(set_to_none=True), step=lambda: None)
    module.manual_backward = lambda loss, **kw: loss.backward(**kw)
    return module


def run(config_key, batch_size, seq_len, steps=3):
    tb = CONFIGS['flexllama'][config_key]
    model_config = tb.model_config
    ctx = tb.training_context

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Build the student from config dims (skip warm-start download - peak memory
    # depends on the shapes, not the weight values).
    trainer = tb.training_method(model_config, ctx)
    trainer = instrument(trainer)
    trainer.cuda().train()
    trainer._ensure_teacher()
    if getattr(ctx, 'activation_checkpointing', False):
        from training_memopt import enable_activation_checkpointing
        n = enable_activation_checkpointing(trainer.submodel)
        print(f"    activation checkpointing on {n} blocks", flush=True)

    opt = torch.optim.AdamW(trainer.submodel.parameters(), lr=1e-5)
    trainer.optimizers = lambda: types.SimpleNamespace(
        zero_grad=lambda: opt.zero_grad(set_to_none=True), step=opt.step)

    after_load = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()

    ok, err = True, None
    try:
        for _ in range(steps):
            ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                trainer._step((ids, ids), "train")
            torch.cuda.synchronize()
    except torch.OutOfMemoryError as e:
        ok, err = False, str(e).split('\n')[0]

    peak = torch.cuda.max_memory_allocated() / 2**30
    del trainer, opt
    torch.cuda.empty_cache()
    return ok, after_load, peak, err


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"capacity: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB\n", flush=True)

    seq_len = 1024
    cases = [
        # (label, config_key, batch_size)
        ("original DDP trainer, bs=4", 'fineweb.kd_qwen25_1p5b_warmstart', 4),
        ("original DDP trainer, bs=2", 'fineweb.kd_qwen25_1p5b_warmstart', 2),
        ("memopt trainer,       bs=4", 'fineweb.kd_qwen25_1p5b_warmstart_memopt', 4),
        ("memopt trainer,       bs=8", 'fineweb.kd_qwen25_1p5b_warmstart_memopt', 8),
    ]

    print(f"{'case':<32} {'result':<8} {'model+teacher':>14} {'peak':>10}")
    print("-" * 70)
    for label, key, bs in cases:
        print(f"  running {label} ...", flush=True)
        ok, after_load, peak, err = run(key, bs, seq_len)
        status = "OK" if ok else "OOM"
        print(f"{label:<32} {status:<8} {after_load:>11.2f} GiB {peak:>7.2f} GiB", flush=True)
        if err:
            print(f"    -> {err}", flush=True)
    print("\ndone")
