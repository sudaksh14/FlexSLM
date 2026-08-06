"""Numerical equivalence: memopt trainers vs the originals in training.py."""
import types
import torch
from torch.utils.data import DataLoader, TensorDataset

from networks.flexllama import FlexLLaMAConfig
from training import FlexLMTrainer, FlexLMKDTrainer
from training_memopt import MemOptFlexLMTrainer, MemOptFlexLMKDTrainer, enable_activation_checkpointing
from config.experiments import LLaMATrainingContext, LLaMAKDTrainingContext

VOCAB = 512


def make_config():
    return FlexLLaMAConfig(
        vocab_size=VOCAB, max_seq_length=32, num_layers=3,
        hidden_dims=(32, 64, 96), num_heads=(4, 8, 12), num_kv_heads=(4, 8, 12),
        intermediate_dims=(64, 128, 192), tie_embeddings=False,
    )


def dummy_loader():
    x = torch.randint(0, VOCAB, (8, 16))
    ds = TensorDataset(x, x)
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def instrument(module):
    """Stub out the Lightning plumbing _step relies on; record logged scalars."""
    logged = {}
    module.log = lambda name, val, **kw: logged.__setitem__(
        name, float(val.detach()) if torch.is_tensor(val) else float(val))
    module.optimizers = lambda: types.SimpleNamespace(
        zero_grad=lambda: module.zero_grad(set_to_none=True), step=lambda: None)
    module.manual_backward = lambda loss, **kw: loss.backward(**kw)
    return logged


def build_pair(trainer_cls_a, trainer_cls_b, ctx_a, ctx_b, kd, seed=0):
    torch.manual_seed(seed)
    a = trainer_cls_a(make_config(), ctx_a)
    torch.manual_seed(seed)
    b = trainer_cls_b(make_config(), ctx_b)
    b.load_state_dict(a.state_dict())  # identical weights, not just identical seed
    if kd:
        torch.manual_seed(123)
        teacher = torch.nn.Sequential(torch.nn.Embedding(VOCAB, 32), torch.nn.Linear(32, VOCAB))
        wrapped = lambda ids, _t=teacher: types.SimpleNamespace(logits=_t(ids))
        for m in (a, b):
            object.__setattr__(m, '_teacher', wrapped)
    return a, b


def compare(a, b, batch, stage, label):
    la, lb = instrument(a), instrument(b)
    a.train(); b.train()
    a._step(batch, stage)
    b._step(batch, stage)

    assert set(la) == set(lb), f"logged metric names differ: {set(la) ^ set(lb)}"
    # relative tolerance: chunk-wise summation reorders float adds, and ppl=exp(loss)
    # amplifies that into the absolute value, so compare relatively.
    def rel(x, y):
        return abs(x - y) / max(1.0, abs(x), abs(y))
    worst_loss = max(rel(la[k], lb[k]) for k in la)
    for k in sorted(la):
        assert rel(la[k], lb[k]) < 1e-5, f"{k}: {la[k]} vs {lb[k]} (rel {rel(la[k], lb[k]):.2e})"

    worst_grad, n_grads = 0.0, 0
    if stage == "train":
        ga = dict(a.named_parameters())
        gb = dict(b.named_parameters())
        for name in ga:
            if ga[name].grad is None and gb[name].grad is None:
                continue
            assert ga[name].grad is not None and gb[name].grad is not None, f"{name}: grad presence differs"
            d = (ga[name].grad - gb[name].grad).abs().max().item()
            worst_grad = max(worst_grad, d)
            n_grads += 1
    print(f"  {label:<34} metrics={len(la):>2}  max|Δloss|={worst_loss:.3e}  "
          f"grads={n_grads:>3}  max|Δgrad|={worst_grad:.3e}")
    return worst_loss, worst_grad


torch.manual_seed(7)
batch_ids = torch.randint(0, VOCAB, (4, 16))
batch = (batch_ids, batch_ids)

print("\n=== non-KD: FlexLMTrainer vs MemOptFlexLMTrainer ===")
for chunk in (16, 999999):  # many chunks, and a single chunk
    ctx_a = LLaMATrainingContext(epochs=1, wandb_project_name=None)
    ctx_a.loader_function = dummy_loader
    ctx_b = LLaMATrainingContext(epochs=1, wandb_project_name=None)
    ctx_b.loader_function = dummy_loader
    ctx_b.loss_chunk_size = chunk
    a, b = build_pair(FlexLMTrainer, MemOptFlexLMTrainer, ctx_a, ctx_b, kd=False)
    compare(a, b, batch, "train", f"chunk={chunk}")

print("\n=== KD: FlexLMKDTrainer vs MemOptFlexLMKDTrainer ===")
for chunk in (16, 999999):
    ctx_a = LLaMAKDTrainingContext(kd_lambda=0.5, kd_temperature=2.0, epochs=1, wandb_project_name=None)
    ctx_a.loader_function = dummy_loader
    ctx_b = LLaMAKDTrainingContext(kd_lambda=0.5, kd_temperature=2.0, epochs=1, wandb_project_name=None)
    ctx_b.loader_function = dummy_loader
    ctx_b.loss_chunk_size = chunk
    a, b = build_pair(FlexLMKDTrainer, MemOptFlexLMKDTrainer, ctx_a, ctx_b, kd=True)
    compare(a, b, batch, "train", f"chunk={chunk}")

print("\n=== KD with a non-default kd_lambda (1.0 = pure KL) ===")
ctx_a = LLaMAKDTrainingContext(kd_lambda=1.0, kd_temperature=3.0, epochs=1, wandb_project_name=None)
ctx_a.loader_function = dummy_loader
ctx_b = LLaMAKDTrainingContext(kd_lambda=1.0, kd_temperature=3.0, epochs=1, wandb_project_name=None)
ctx_b.loader_function = dummy_loader
ctx_b.loss_chunk_size = 16
a, b = build_pair(FlexLMKDTrainer, MemOptFlexLMKDTrainer, ctx_a, ctx_b, kd=True)
compare(a, b, batch, "train", "kd_lambda=1.0 chunk=16")

print("\n=== validation stage (no_grad path) ===")
ctx_a = LLaMAKDTrainingContext(kd_lambda=0.5, kd_temperature=2.0, epochs=1, wandb_project_name=None)
ctx_a.loader_function = dummy_loader
ctx_b = LLaMAKDTrainingContext(kd_lambda=0.5, kd_temperature=2.0, epochs=1, wandb_project_name=None)
ctx_b.loader_function = dummy_loader
ctx_b.loss_chunk_size = 16
a, b = build_pair(FlexLMKDTrainer, MemOptFlexLMKDTrainer, ctx_a, ctx_b, kd=True)
with torch.no_grad():
    compare(a, b, batch, "val", "val chunk=16")

print("\n=== activation checkpointing: same grads as without ===")
ctx_a = LLaMAKDTrainingContext(kd_lambda=0.5, kd_temperature=2.0, epochs=1, wandb_project_name=None)
ctx_a.loader_function = dummy_loader
ctx_b = LLaMAKDTrainingContext(kd_lambda=0.5, kd_temperature=2.0, epochs=1, wandb_project_name=None)
ctx_b.loader_function = dummy_loader
ctx_b.loss_chunk_size = 16
a, b = build_pair(MemOptFlexLMKDTrainer, MemOptFlexLMKDTrainer, ctx_a, ctx_b, kd=True)
n = enable_activation_checkpointing(b.submodel)
print(f"  wrapped {n} decoder blocks with checkpointing")
assert n == 3, n
sd_keys_a, sd_keys_b = set(a.state_dict()), set(b.state_dict())
assert sd_keys_a == sd_keys_b, f"checkpointing changed state_dict keys: {sd_keys_a ^ sd_keys_b}"
print("  state_dict keys unchanged by checkpointing: OK")
compare(a, b, batch, "train", "ckpt on vs off")

print("\nALL EQUIVALENCE CHECKS PASSED")
