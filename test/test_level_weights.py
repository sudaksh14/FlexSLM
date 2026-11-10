"""level_weighting flag (training/level_weights.py and FlexLMSFTTrainer)."""
import math
import types

import pytest
import torch

from networks.flexllama import FlexLLaMAConfig
from training.level_weights import LevelWeighter, uncertainty_grad
from training.sft import FlexLMSFTTrainer, SFTTrainingContext, collate_qa

VOCAB = 128


# ---- LevelWeighter on its own ----

def test_none_is_identity():
    w = LevelWeighter(3, "none")
    for _ in range(5):
        w.observe_step({0: torch.tensor(9.0), 2: torch.tensor(1.0)}, "cpu")
    assert not w.active and [w.weight(k) for k in range(3)] == [1.0, 1.0, 1.0]


def test_uncertainty_grad_matches_autograd():
    for s0, L in ((0.0, 2.5), (1.3, 0.4), (-0.7, 6.0)):
        s = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
        (torch.exp(-s) * L + s).backward()
        assert math.isclose(uncertainty_grad(s0, L), s.grad.item(), rel_tol=1e-12)


def test_uncertainty_converges_to_inverse_loss():
    # Constant per-level losses: s_k -> log L_k, weight -> 1/L_k, so the
    # high-loss (small) level ends up with the smaller weight.
    w = LevelWeighter(3, "uncertainty", lr=1e-2)
    losses = {0: 6.0, 1: 3.0, 2: 1.5}
    for _ in range(3000):
        w.observe_step({k: torch.tensor(v) for k, v in losses.items()}, "cpu")
    for k, L in losses.items():
        assert abs(w.log_vars[k] - math.log(L)) < 0.02
        assert abs(w.weight(k) - 1 / L) / (1 / L) < 0.02
    assert w.weight(0) < w.weight(1) < w.weight(2)


def test_uncertainty_skips_untrained_levels():
    w = LevelWeighter(3, "uncertainty")
    w.observe_step({0: torch.tensor(5.0), 2: torch.tensor(1.0)}, "cpu")
    assert w.log_vars[1] == 0.0 and w._t == [1, 0, 1]
    assert w.log_vars[0] > 0  # L > 1 -> s grows (weight falls)


def test_loss_proportional_favours_high_loss():
    w = LevelWeighter(3, "loss_proportional", ema_decay=0.5)
    assert w.weight(0) == 1.0  # nothing seen yet
    for _ in range(20):
        w.observe_step({0: torch.tensor(6.0), 1: torch.tensor(3.0), 2: torch.tensor(1.5)}, "cpu")
    ws = [w.weight(k) for k in range(3)]
    assert ws[0] > ws[1] > ws[2]
    assert math.isclose(sum(ws) / 3, 1.0, rel_tol=1e-9)


def test_state_dict_roundtrip_and_mode_check():
    a = LevelWeighter(3, "uncertainty")
    for _ in range(7):
        a.observe_step({0: torch.tensor(4.0), 1: torch.tensor(2.0), 2: torch.tensor(1.0)}, "cpu")
    b = LevelWeighter(3, "uncertainty")
    b.load_state_dict(a.state_dict())
    assert b.state_dict() == a.state_dict()
    with pytest.raises(ValueError):
        LevelWeighter(3, "loss_proportional").load_state_dict(a.state_dict())


def test_bad_flag_rejected_by_context():
    with pytest.raises(ValueError):
        SFTTrainingContext(load_from=None, teacher_hf_model="x", tokenizer_name="x", level_weighting="kendall")


# ---- In the SFT trainer: the weight scales exactly that level's gradient ----

def _config():
    return FlexLLaMAConfig(
        vocab_size=VOCAB, max_seq_length=32, num_layers=2,
        hidden_dims=(16, 32, 48), num_heads=(2, 4, 6), num_kv_heads=(2, 4, 6),
        intermediate_dims=(32, 64, 96), tie_embeddings=False)


def _trainer(mode, train_levels):
    ctx = SFTTrainingContext(load_from=None, teacher_hf_model="x", tokenizer_name="x",
                             kd_lambda=0.0, train_levels=train_levels, level_weighting=mode,
                             wandb_project_name=None)
    torch.manual_seed(0)
    t = FlexLMSFTTrainer(_config(), ctx)
    logged = {}
    t.log = lambda name, val, **kw: logged.__setitem__(name, float(val))
    t.optimizers = lambda: types.SimpleNamespace(zero_grad=lambda: t.zero_grad(set_to_none=True), step=lambda: None)
    t.manual_backward = lambda loss, **kw: loss.backward(**kw)
    t.train()
    return t, logged


def _batch():
    torch.manual_seed(1)
    return collate_qa([(torch.randint(0, VOCAB, (6,)).tolist(), torch.randint(0, VOCAB, (5,)).tolist())
                       for _ in range(3)])


def _grads(t):
    return {n: p.grad.clone() for n, p in t.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("level", [0, 2])
def test_weight_scales_level_gradient(level):
    batch = _batch()
    ref, ref_log = _trainer("none", [level])
    ref._step(batch, "train")
    g_ref = _grads(ref)

    t, log = _trainer("uncertainty", [level])
    t.level_weighter.log_vars[level] = math.log(4.0)  # weight 0.25
    t._step(batch, "train")
    g = _grads(t)

    assert g.keys() == g_ref.keys() and g
    for n in g:
        torch.testing.assert_close(g[n], 0.25 * g_ref[n], rtol=1e-5, atol=1e-7)
    # Logged losses stay unweighted.
    assert math.isclose(log[f"train_level{level}_loss"], ref_log[f"train_level{level}_loss"], rel_tol=1e-6)
    assert log[f"train_level{level}_weight"] == pytest.approx(0.25)
    # The log-variance moved after the step (one Adam step of size ~lr).
    assert t.level_weighter.log_vars[level] != math.log(4.0)


def test_none_mode_logs_no_weights():
    t, log = _trainer("none", None)
    t._step(_batch(), "train")
    assert not any(k.endswith("_weight") for k in log)


def test_validation_is_unweighted():
    t, log = _trainer("uncertainty", None)
    t.level_weighter.log_vars = [2.0, 1.0, 0.5]
    before = list(t.level_weighter.log_vars)
    with torch.no_grad():
        t._step(_batch(), "val")
    assert t.level_weighter.log_vars == before
    assert not any(k.endswith("_weight") for k in log)


# ---- fixed and dwa modes ----

def test_fixed_weights_and_validation():
    w = LevelWeighter(3, "fixed", fixed=(3.0, 1.0, 1.0))
    assert [w.weight(k) for k in range(3)] == [3.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        LevelWeighter(3, "fixed", fixed=(1.0, 1.0))
    with pytest.raises(ValueError):
        LevelWeighter(3, "fixed")


def test_dwa_favours_slowest_falling_level():
    w = LevelWeighter(3, "dwa", ema_decay=0.0, dwa_window=5, dwa_temperature=2.0)
    for step in range(5):  # not enough history yet
        w.observe_step({0: torch.tensor(6.0), 1: torch.tensor(4.0 - 0.3 * step), 2: torch.tensor(2.0 - 0.2 * step)}, "cpu")
        assert [w.weight(k) for k in range(3)] == [1.0, 1.0, 1.0]
    w.observe_step({0: torch.tensor(6.0), 1: torch.tensor(2.5), 2: torch.tensor(1.0)}, "cpu")
    ws = [w.weight(k) for k in range(3)]
    assert ws[0] > ws[1] and ws[0] > ws[2]          # level 0's loss stopped falling
    assert math.isclose(sum(ws), 3.0, rel_tol=1e-9)  # K * softmax


def test_dwa_state_roundtrip():
    a = LevelWeighter(3, "dwa", dwa_window=3)
    for s in range(6):
        a.observe_step({k: torch.tensor(5.0 - s * (k + 1) * 0.1) for k in range(3)}, "cpu")
    b = LevelWeighter(3, "dwa", dwa_window=3)
    b.load_state_dict(a.state_dict())
    assert [b.weight(k) for k in range(3)] == [a.weight(k) for k in range(3)]


# ---- single-level sampling and the shared schedule ----

def test_single_level_sampling_is_step_seeded():
    ctx = SFTTrainingContext(load_from=None, teacher_hf_model="x", tokenizer_name="x",
                             kd_lambda=0.0, level_sampling="single", wandb_project_name=None)
    torch.manual_seed(0)
    t = FlexLMSFTTrainer(_config(), ctx)
    seen = []
    for step in range(60):
        type(t).global_step = property(lambda self, s=step: s)
        a, b = t._select_levels("train"), t._select_levels("train")
        assert a == b and len(a) == 1  # same draw every call (and so every rank) at a step
        seen += a
    assert set(seen) == {0, 1, 2}
    assert t._select_levels("val") == [0, 1, 2]
    del type(t).global_step  # restore Lightning's property
    with pytest.raises(ValueError):
        SFTTrainingContext(load_from=None, teacher_hf_model="x", tokenizer_name="x", level_sampling="matformer")


def test_warmup_cosine_matches_previous_sft_schedule():
    from training.trainer import warmup_cosine_scheduler
    total, ratio = 1044, 0.01
    warmup = math.ceil(ratio * total)
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = warmup_cosine_scheduler(opt, total, ratio)
    for step in range(total):
        expect = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
        assert math.isclose(opt.param_groups[0]["lr"], expect, abs_tol=1e-12)
        opt.step(); sched.step()


# ---- Stage-1 (memopt KD) trainer ----

def _kd_trainer(mode, **kw):
    from training.stage1 import FlexLMStage1Trainer, Stage1Context
    ctx = Stage1Context(train_tokens=1000, level_weighting=mode, wandb_project_name=None, **kw)
    torch.manual_seed(0)
    t = FlexLMStage1Trainer(_config(), ctx)
    torch.manual_seed(5)
    teacher = torch.nn.Sequential(torch.nn.Embedding(VOCAB, 16), torch.nn.Linear(16, VOCAB))
    object.__setattr__(t, "_teacher", lambda ids: types.SimpleNamespace(logits=teacher(ids)))
    logged = {}
    t.log = lambda name, val, **kw: logged.__setitem__(name, float(val))
    t.optimizers = lambda: types.SimpleNamespace(zero_grad=lambda: t.zero_grad(set_to_none=True), step=lambda: None)
    t.lr_schedulers = lambda: types.SimpleNamespace(step=lambda: None)
    t.manual_backward = lambda loss, **kw: loss.backward(**kw)
    t.train()
    return t, logged


def test_stage1_weighting_scales_gradients():
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (2, 12))
    ref, ref_log = _kd_trainer("none")
    ref._step((ids, ids), "train")
    t, log = _kd_trainer("fixed", level_weights_fixed=(2.0, 2.0, 2.0))
    t._step((ids, ids), "train")
    g_ref, g = _grads(ref), _grads(t)
    assert g.keys() == g_ref.keys() and g
    for n in g:
        torch.testing.assert_close(g[n], 2.0 * g_ref[n], rtol=1e-5, atol=1e-7)
    for k in range(3):
        assert math.isclose(log[f"train_level{k}_loss"], ref_log[f"train_level{k}_loss"], rel_tol=1e-6)


def test_pmc_memmap_loader(tmp_path):
    import json
    import numpy as np
    from training.stage1 import load_pmc_memmap
    for split, n in (("train", 50_000), ("val", 5_000), ("test", 5_000)):
        np.arange(n, dtype=np.uint32).tofile(tmp_path / f"{split}.bin")
    (tmp_path / "meta.json").write_text(json.dumps(dict(dtype="uint32", train_tokens=50_000)))
    train, val, test = load_pmc_memmap(str(tmp_path), max_seq_length=64, batch_size=4,
                                       train_tokens=6_400, val_batches=2, test_batches=3, num_workers=0)
    assert len(train.dataset) == 100 and len(val.dataset) == 8 and len(test.dataset) == 12
    x, y = next(iter(test))
    assert x.shape == (4, 64) and torch.equal(x, y)
    assert int(x.max()) < 5_000  # test windows come from test.bin
    with pytest.raises(FileNotFoundError):
        load_pmc_memmap(str(tmp_path / "missing"))


# ---- the registered paper configs ----

def test_stage1_and_stage2_configs_register():
    from training.run_experiment import resolve_from_str
    from training.stage1 import S1_FULL_TOKENS, S1_PILOT_ARMS, S1_PILOT_TOKENS, FlexLMStage1Trainer
    for arm, overrides in S1_PILOT_ARMS:
        b = resolve_from_str(f"flexllama,s1.qwen3_0p6b.pilot.{arm}")
        assert b.training_method is FlexLMStage1Trainer
        c = b.training_context
        assert c.train_tokens == S1_PILOT_TOKENS and c.teacher_hf_model == "Qwen/Qwen3-1.7B"
        for k, v in overrides.items():
            if k != "ordering":  # model-side (start weights), checked below
                assert getattr(c, k) == v, (arm, k)
        assert list(b.model_config.hidden_dims) == [512, 768, 1024]
    assert resolve_from_str("flexllama,s1.qwen3_0p6b.full").training_context.train_tokens == S1_FULL_TOKENS

    expect = {"": "none", ".w_fixed": "fixed", ".w_uncertainty": "uncertainty",
              ".w_lossprop": "loss_proportional", ".w_dwa": "dwa"}
    for suffix, mode in expect.items():
        c = resolve_from_str(f"flexllama,sft.paper.qwen3_0p6b_nested{suffix}").training_context
        assert c.level_weighting == mode
        assert c.load_from == "flexllama,s1.qwen3_0p6b.full" and c.kd_lambda == 0.5
        assert c.teacher_hf_model.endswith("level2") and c.lr == 5e-5 and c.train_levels is None
    assert resolve_from_str("flexllama,sft.paper.qwen3_0p6b_nested.single").training_context.level_sampling == "single"
    assert resolve_from_str("flexllama,sft.paper.qwen3_0p6b_nested.no_kd").training_context.kd_lambda == 0.0
    no_s1 = resolve_from_str("flexllama,sft.paper.qwen3_0p6b_nested.no_stage1")
    assert no_s1.training_context.load_from is None
    from training.importance_order import FINAL_ORDERING, ORDERED_CHECKPOINTS, start_weights
    final_start = start_weights(FINAL_ORDERING)
    assert no_s1.model_config.pretrained_hf_model == final_start  # final ordering, only Stage 1 removed
    assert no_s1.model_config.max_seq_length >= 3072
    # M1 is a pilot dimension; the final run follows FINAL_ORDERING.
    assert resolve_from_str("flexllama,s1.qwen3_0p6b.full").model_config.pretrained_hf_model == final_start
    starts = {arm: resolve_from_str(f"flexllama,s1.qwen3_0p6b.pilot.{arm}").model_config.pretrained_hf_model
              for arm in ("base", "order_hidden_act", "order_taylor", "lr2e-4")}
    assert starts == {"base": "Qwen/Qwen3-0.6B", "order_hidden_act": ORDERED_CHECKPOINTS["hidden_act"],
                      "order_taylor": ORDERED_CHECKPOINTS["taylor"], "lr2e-4": "Qwen/Qwen3-0.6B"}
    a1 = resolve_from_str("flexllama,s1.qwen3_0p6b.full.no_order")
    assert a1.model_config.pretrained_hf_model == "Qwen/Qwen3-0.6B" and a1.training_context.train_tokens == S1_FULL_TOKENS
    assert (resolve_from_str("flexllama,sft.paper.qwen3_0p6b_nested.no_order").training_context.load_from
            == "flexllama,s1.qwen3_0p6b.full.no_order")
    mf = resolve_from_str("flexllama,sft.paper.qwen3_0p6b_matformer")
    assert (mf.training_context.level_sampling, mf.training_context.kd_lambda, mf.training_context.load_from,
            mf.model_config.pretrained_hf_model) == ("single", 0.0, None, "Qwen/Qwen3-0.6B")
    # Micro-batching: 8 per pass in every paper SFT run, 4 in Stage 1.
    assert resolve_from_str("flexllama,sft.paper.qwen3_1p7b_teacher").training_context.micro_batch_size == 8
    assert resolve_from_str("flexllama,s1.qwen3_0p6b.full").training_context.micro_batch_size == 4


@pytest.mark.parametrize("micro", [1, 2])
def test_micro_batches_match_one_pass(micro):
    # Uneven prompt/answer lengths, so micro-batches hold different token counts.
    torch.manual_seed(2)
    batch = collate_qa([(torch.randint(0, VOCAB, (n,)).tolist(), torch.randint(0, VOCAB, (m,)).tolist())
                        for n, m in [(3, 7), (8, 2), (5, 5), (2, 9)]])
    ref, ref_log = _trainer("none", None)
    ref._step(batch, "train")
    t, log = _trainer("none", None)
    t.training_context.micro_batch_size = micro
    t._step(batch, "train")
    g_ref, g = _grads(ref), _grads(t)
    assert g.keys() == g_ref.keys() and g
    for n in g:
        torch.testing.assert_close(g[n], g_ref[n], rtol=1e-4, atol=1e-6)
    for k, v in ref_log.items():
        assert log[k] == pytest.approx(v, rel=1e-5, abs=1e-7), k


def test_micro_batch_size_validated():
    with pytest.raises(ValueError):
        SFTTrainingContext(load_from=None, teacher_hf_model="x", tokenizer_name="x",
                           micro_batch_size=0, wandb_project_name=None)


@pytest.mark.parametrize("micro", [1, 3])
def test_stage1_micro_batches_match_one_pass(micro):
    torch.manual_seed(4)
    ids = torch.randint(0, VOCAB, (4, 12))
    ref, ref_log = _kd_trainer("none", micro_batch_size=None)
    ref._step((ids, ids), "train")
    t, log = _kd_trainer("none", micro_batch_size=micro)
    t._step((ids, ids), "train")
    g_ref, g = _grads(ref), _grads(t)
    assert g.keys() == g_ref.keys() and g
    for n in g:
        torch.testing.assert_close(g[n], g_ref[n], rtol=1e-4, atol=1e-6)
    for k, v in ref_log.items():
        assert log[k] == pytest.approx(v, rel=1e-5, abs=1e-7), k
