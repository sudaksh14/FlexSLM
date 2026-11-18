"""Stage-1 KD-weight sweep (2026-11-18): HydraLLM's objective CE + lambda KL (kd_form "additive"),
HydraLLM's weighted one-width-per-step sampling, the sweep / teacher-ablation configs and their Stage 2,
and the eval_gen additions for the inference-compute curve (think prompt, budgets, subset)."""
import collections
import random
import types

import pytest
import torch

from networks.flexllama import FlexLLaMAConfig

V = 97


def test_kd_mix_forms():
    from training.trainer import kd_mix
    ce, kl = torch.tensor(2.0), torch.tensor(4.0)
    assert kd_mix(ce, kl, 0.25) == 0.75 * 2 + 0.25 * 4          # convex, the default
    assert kd_mix(ce, kl, 0.25, "additive") == 2 + 0.25 * 4
    assert kd_mix(ce, kl, 0.0, "additive") == kd_mix(ce, kl, 0.0, "convex") == 2
    with pytest.raises(ValueError):
        kd_mix(ce, kl, 0.1, "mixed")


def _fake(weights, sampling="single", levels=9):
    return types.SimpleNamespace(
        submodel=types.SimpleNamespace(max_level=lambda: levels - 1),
        training_context=types.SimpleNamespace(level_sampling=sampling, level_sampling_weights=weights,
                                               num_levels_per_step=None, anchor_levels=None, seed=0),
        global_step=0)


def test_hydra_sampling_one_width_per_step_full_width_three_times_as_often():
    from training.stage1 import HYDRA_WEIGHTS_9
    from training.trainer import FlexLMTrainer
    fake = _fake(HYDRA_WEIGHTS_9)
    counts = collections.Counter()
    n = 22_000
    for step in range(n):
        fake.global_step = step
        lv = FlexLMTrainer._select_levels(fake, "train")
        assert len(lv) == 1
        counts[lv[0]] += 1
    assert set(counts) == set(range(9))
    assert abs(counts[8] / n - 3 / 11) < 0.015
    for i in range(8):
        assert abs(counts[i] / n - 1 / 11) < 0.01
    assert 2.6 < counts[8] / (sum(counts[i] for i in range(8)) / 8) < 3.4
    # deterministic in (seed, step): every DDP rank trains the same width
    fake.global_step = 123
    assert FlexLMTrainer._select_levels(fake, "train") == FlexLMTrainer._select_levels(fake, "train")
    assert FlexLMTrainer._select_levels(fake, "val") == list(range(9))


def test_uniform_single_sampling_is_unchanged():
    # Runs before 2026-11-18 (MatFormer baseline, A5) drew rng.choice: same draws as before.
    from training.trainer import FlexLMTrainer
    fake = _fake(None, levels=3)
    for step in range(30):
        fake.global_step = step
        assert FlexLMTrainer._select_levels(fake, "train") == [random.Random(step).choice([0, 1, 2])]


def test_sampling_weights_must_match_levels():
    from training.trainer import FlexLMTrainer
    with pytest.raises(ValueError):
        FlexLMTrainer._select_levels(_fake((1.0, 3.0), levels=3), "train")


def test_memopt_additive_equals_rescaled_convex():
    # CE + 0.25 KL = (0.8 CE + 0.2 KL) / 0.8: the gradients must be exactly that ratio.
    from config.experiments import LLaMAKDTrainingContext
    from training.memopt import MemOptFlexLMKDTrainer
    cfg = FlexLLaMAConfig(vocab_size=V, max_seq_length=32, num_layers=2, hidden_dims=(32, 64),
                          num_heads=(4, 8), num_kv_heads=(4, 8), intermediate_dims=(64, 128))
    ids = torch.randint(0, V, (4, 16), generator=torch.Generator().manual_seed(2))
    t_logits = torch.randn(4, 16, V, generator=torch.Generator().manual_seed(3))
    models = []
    for lam, form in ((0.25, "additive"), (0.2, "convex")):
        torch.manual_seed(0)
        ctx = LLaMAKDTrainingContext(kd_lambda=lam, kd_temperature=2.0, epochs=1, wandb_project_name=None)
        ctx.kd_form = form
        m = MemOptFlexLMKDTrainer(cfg, ctx)
        object.__setattr__(m, "_teacher", lambda x: types.SimpleNamespace(logits=t_logits[:x.shape[0]]))
        m.log = lambda *a, **k: None
        m.optimizers = lambda m=m: types.SimpleNamespace(zero_grad=lambda: m.zero_grad(set_to_none=True),
                                                         step=lambda: None)
        m.manual_backward = lambda loss, **kw: loss.backward(**kw)
        m._optimizer_step = lambda opt: None
        m.train()
        models.append(m)
    models[1].load_state_dict(models[0].state_dict())
    for m in models:
        m._step((ids, ids), "train")
    n = 0
    for (name, pa), (_, pb) in zip(models[0].named_parameters(), models[1].named_parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad / 0.8, msg=name, rtol=1e-4, atol=1e-6)
            n += 1
    assert n > 0


def test_kd_sweep_configs_are_the_final_pilot_recipe_plus_kd():
    from training.run_experiment import resolve_from_str
    from training.stage1 import (CORRECTED_TEACHER_1P7B, HYDRA_WEIGHTS_9, S1_KD_PILOTS, S1_PILOT_TOKENS,
                                 kd_pilot_name)
    base = resolve_from_str("flexllama,s1.qwen3_0p6b.pilot.final")
    assert len(S1_KD_PILOTS) == 12
    for t, teacher in (("corr", CORRECTED_TEACHER_1P7B), ("orig", "Qwen/Qwen3-1.7B")):
        for lam in (0.1, 0.3, 0.5):
            for hydra in (False, True):
                b = resolve_from_str(f"flexllama,{kd_pilot_name(lam, t, hydra)}")
                c = b.training_context
                assert (c.kd_lambda, c.kd_form, c.kd_temperature, c.teacher_hf_model) == (lam, "additive", 2.0, teacher)
                assert c.train_tokens == S1_PILOT_TOKENS
                for k in ("lr", "weight_decay", "gradient_clip_val", "seed", "warmup_ratio", "micro_batch_size"):
                    assert getattr(c, k) == getattr(base.training_context, k), k
                assert b.model_config == base.model_config  # same grid, same Minitron-ordered warm start
                if hydra:
                    assert (c.level_sampling, c.level_sampling_weights) == ("single", HYDRA_WEIGHTS_9)
                    assert c.anchor_levels is None and c.num_levels_per_step is None
                else:
                    assert c.level_sampling == "sandwich" and c.level_sampling_weights is None
                    assert (c.anchor_levels, c.num_levels_per_step) == ((0, 4, 8), 4)
    assert kd_pilot_name(0.1, "corr") == "s1.qwen3_0p6b.pilot.final.kdadd0p1.corr"
    # every older config keeps the convex form
    assert base.training_context.kd_form == "convex"


def test_kd_sweep_stage2_is_the_final9_stage2():
    from training.run_experiment import resolve_from_str
    from training.stage1 import S1_KD_PILOTS
    ref = resolve_from_str("flexllama,sft.pilot.qwen3_0p6b_final9").training_context
    bs = lambda ctx: ctx.loader_function.keywords["batch_size"]
    for s1 in S1_KD_PILOTS:
        suffix = s1[len("s1.qwen3_0p6b.pilot.final"):]
        for gpu in ("", ".1gpu"):
            c = resolve_from_str(f"flexllama,sft.pilot.qwen3_0p6b_final9{suffix}{gpu}").training_context
            assert c.load_from == f"flexllama,{s1}"
            for k in ("teacher_hf_model", "kd_lambda", "kd_temperature", "lr", "anchor_levels", "num_levels_per_step",
                      "micro_batch_size", "gradient_clip_val", "seed", "level_sampling"):
                assert getattr(c, k) == getattr(ref, k), (s1, k)
            assert bs(c) == (32 if gpu else bs(ref))


def test_think_prompt_enables_thinking():
    from eval.eval_gen import render_prompt
    stored = "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert render_prompt(stored, "think") == "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"
    with pytest.raises(ValueError):
        render_prompt("<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n", "think")


def test_trim_keeps_the_first_stop_token_and_drops_padding():
    from eval.eval_gen import trim
    assert trim([5, 6, 7, 1, 0, 0], {1, 0}) == [5, 6, 7, 1]
    assert trim([5, 6, 7], {1}) == [5, 6, 7]


def test_stage2_only_ablations_change_one_thing_each():
    # 2026-11-18: the paper's method is sft.paper.qwen3_0p6b_nested.no_stage1 (A2); each ablation changes one field.
    from training.run_experiment import resolve_from_str
    base = resolve_from_str("flexllama,sft.paper.qwen3_0p6b_nested.no_stage1")
    b = base.training_context
    keys = ("teacher_hf_model", "kd_lambda", "kd_temperature", "lr", "level_sampling", "train_levels", "seed",
            "gradient_clip_val", "load_from")
    expect = {"no_kd": {"kd_lambda": 0.0}, "no_order": {}, "single": {"level_sampling": "single"},
              "indep_l0": {"train_levels": [0]}, "indep_l1": {"train_levels": [1]}, "indep_l2": {"train_levels": [2]}}
    for tag, diff in expect.items():
        m = resolve_from_str(f"flexllama,sft.paper.qwen3_0p6b_nested.no_stage1.{tag}")
        c = m.training_context
        for k in keys:
            want = diff.get(k, getattr(b, k))
            got = getattr(c, k)
            assert (list(got) if isinstance(got, tuple) else got) == (list(want) if isinstance(want, tuple) else want), (tag, k)
        assert c.load_from is None and c.micro_batch_size == 4 and b.micro_batch_size == 8
        if tag == "no_order":
            assert m.model_config.pretrained_hf_model == "Qwen/Qwen3-0.6B" != base.model_config.pretrained_hf_model
        else:
            assert m.model_config == base.model_config
    big = resolve_from_str("flexllama,sft.paper.qwen3_1p7b_nested.no_stage1")
    bc = big.training_context
    assert bc.zero_optimizer and bc.activation_checkpointing and bc.loader_function.keywords["batch_size"] * 4 == 32
    assert big.training_context.kd_lambda == 0.0 and list(big.model_config.hidden_dims) == [1024, 1536, 2048]
    assert big.model_config.pretrained_hf_model.endswith("qwen3-1.7b.minitron")
