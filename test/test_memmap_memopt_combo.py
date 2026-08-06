"""
Compatibility check: memmap data pipeline + memory-optimized trainer together.

The two are orthogonal (loader vs trainer class), but this actually runs a
MemOptFlexLMKDTrainer training step against memmap-backed batches rather than
assuming the interfaces line up.
"""
import json
import os
import tempfile
import types

import numpy as np
import torch

import utils
from utils import load_fineweb_edu_memmap
from networks.flexllama import FlexLLaMAConfig
from training_memopt import MemOptFlexLMKDTrainer, enable_activation_checkpointing
from config.experiments import CONFIGS, LLaMAKDTrainingContext

VOCAB = 32000


def build_fake(dirpath, n_train, n_val):
    os.makedirs(dirpath, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        rng.integers(0, VOCAB, size=n, dtype=np.uint16).tofile(os.path.join(dirpath, f"{split}.bin"))
    with open(os.path.join(dirpath, "meta.json"), "w") as f:
        json.dump({"tokenizer": "fake", "vocab_size": VOCAB, "dtype": "uint16",
                   "train_tokens": n_train, "val_tokens": n_val}, f)


print("=== 1. the shipped combo configs use BOTH the memopt trainer and memmap loader ===")
combo = ['fineweb_mm.kd_qwen25_1p5b_warmstart_memopt',
         'fineweb_mm.kd_smollm2_1p7b_warmstart_memopt']
for k in combo:
    tb = CONFIGS['flexllama'][k]
    tc = tb.training_context
    assert tb.training_method is MemOptFlexLMKDTrainer, f"{k}: {tb.training_method.__name__}"
    assert tb.training_context.loader_function.func is utils.load_fineweb_edu_memmap
    assert getattr(tc, 'loss_chunk_size', None) == 512
    assert getattr(tc, 'activation_checkpointing', False) is True
    print(f"  {k}: trainer=MemOptFlexLMKDTrainer loader=memmap chunk=512 ckpt=True  OK")

print("\n=== 2. memmap batches feed a memopt training step end to end ===")
with tempfile.TemporaryDirectory() as d:
    build_fake(d, n_train=4_000_000, n_val=200_000)
    train, val, test = load_fineweb_edu_memmap(
        max_seq_length=128, batch_size=4, num_workers=2, data_dir=d,
        val_batches=4, test_batches=4)

    model_config = FlexLLaMAConfig(
        vocab_size=VOCAB, max_seq_length=128, num_layers=3,
        hidden_dims=(32, 64, 96), num_heads=(4, 8, 12), num_kv_heads=(4, 8, 12),
        intermediate_dims=(64, 128, 192), tie_embeddings=False,
    )
    ctx = LLaMAKDTrainingContext(
        dataset="fineweb-edu-memmap", kd_lambda=0.5, kd_temperature=2.0,
        epochs=1, patience=1, wandb_project_name=None)
    ctx.loss_chunk_size = 64
    ctx.activation_checkpointing = True

    trainer = MemOptFlexLMKDTrainer(model_config, ctx)
    # small stand-in teacher with the same vocab
    torch.manual_seed(0)
    teacher = torch.nn.Sequential(torch.nn.Embedding(VOCAB, 32), torch.nn.Linear(32, VOCAB))
    object.__setattr__(trainer, '_teacher',
                       lambda ids, _t=teacher: types.SimpleNamespace(logits=_t(ids)))

    n_ckpt = enable_activation_checkpointing(trainer.submodel)
    print(f"  activation checkpointing wrapped {n_ckpt} blocks")

    logged = {}
    trainer.log = lambda name, val, **kw: logged.__setitem__(
        name, float(val.detach()) if torch.is_tensor(val) else float(val))
    opt = torch.optim.AdamW(trainer.submodel.parameters(), lr=1e-4)
    trainer.optimizers = lambda: types.SimpleNamespace(
        zero_grad=lambda: opt.zero_grad(set_to_none=True), step=opt.step)
    trainer.manual_backward = lambda loss, **kw: loss.backward(**kw)
    trainer.train()

    before = trainer.submodel.embed_tokens.embedding.weight.detach().clone()
    for i, batch in enumerate(train):
        assert isinstance(batch, (list, tuple)) and len(batch) == 2, type(batch)
        x, y = batch
        assert x.shape == (4, 128) and torch.equal(x, y), (x.shape,)
        trainer._step(batch, "train")
        if i >= 2:
            break
    after = trainer.submodel.embed_tokens.embedding.weight
    assert not torch.equal(before, after), "weights did not change - optimizer step never took effect"
    assert logged, "no metrics logged"
    assert any(k.endswith('_kl_loss') for k in logged), f"KD metrics missing: {sorted(logged)}"
    assert all(np.isfinite(v) for v in logged.values()), f"non-finite metric: {logged}"
    print(f"  3 training steps OK; {len(logged)} metrics logged, all finite; weights updated")
    print(f"  sample metrics: train_loss={logged.get('train_loss'):.4f} "
          f"level0_kl={logged.get('train_level0_kl_loss'):.4f}")

    print("\n=== 3. validation path (no_grad) also works with memmap batches ===")
    with torch.no_grad():
        trainer.eval()
        vlogged = {}
        trainer.log = lambda name, val, **kw: vlogged.__setitem__(
            name, float(val.detach()) if torch.is_tensor(val) else float(val))
        for batch in val:
            trainer._step(batch, "val")
            break
    assert vlogged and all(np.isfinite(v) for v in vlogged.values()), vlogged
    print(f"  val step OK; {len(vlogged)} metrics, all finite")

print("\n=== 4. map-style loader gives finetune() a real length (exact ckpt cadence) ===")
with tempfile.TemporaryDirectory() as d:
    build_fake(d, n_train=4_000_000, n_val=100_000)
    train, _, _ = load_fineweb_edu_memmap(
        max_seq_length=1024, batch_size=8, num_workers=0, data_dir=d,
        val_batches=2, test_batches=2)
    n = len(train)  # streaming loader raises TypeError here and falls back to an estimate
    print(f"  len(train_loader) = {n} -> val_check_interval = {max(1, n // 20)} (exact, not estimated)")
    assert n == (4_000_000 // 1024) // 8

print("\nMEMMAP + MEMOPT ARE COMPATIBLE")
