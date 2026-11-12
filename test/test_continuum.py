"""Continuum-of-widths machinery: FlexLLaMA.set_widths, per-width norm offsets,
LLaMaFlex modulation (+ export folding), anchor sampling, continuum param groups,
eval/eval_continuum helpers. Tiny random Qwen3-shaped models on CPU."""
import dataclasses
import types

import pytest
import torch

from config.experiments import _flex_llama_dims_fixed_head
from networks.flexllama import FlexLLaMAConfig

H, NH, NKV, I, HD, L, V = 256, 8, 4, 384, 16, 3, 97


def _cfg(fractions=(0.5, 0.75, 1.0), **kw):
    hd, hs, kvs, im = _flex_llama_dims_fixed_head(H, NH, NKV, I, fractions=fractions, head_dim=HD)
    return FlexLLaMAConfig(vocab_size=V, max_seq_length=32, num_layers=L, hidden_dims=hd, num_heads=hs,
                           num_kv_heads=kvs, intermediate_dims=im, tie_embeddings=True, qkv_bias=False,
                           qk_norm=True, head_dim=HD, **kw)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    m = cfg.make_model().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.uniform_(0.5, 1.5)
            elif "level_delta" not in n and "_mod." not in n:
                p.normal_(0, 0.05)
    return m


def _x():
    return torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(1))


def _no_rope(sd):
    return {k: v for k, v in sd.items() if ".cos_" not in k and ".sin_" not in k}


def test_set_widths_matches_trained_levels_and_clears():
    cfg = _cfg()
    m, x = _model(cfg), _x()
    for lvl in range(3):
        m.set_level_use(lvl)
        ref = m(x)
        m.set_widths(cfg.hidden_dims[lvl], cfg.intermediate_dims[lvl], cfg.num_heads[lvl])
        torch.testing.assert_close(m(x), ref)
    m.set_level_use(0)
    assert all(getattr(mod, "dims_override", None) is None and getattr(mod, "dim_override", None) is None
               for mod in m.modules())
    assert all(b.self_attn.heads_override is None for b in m.layers)


def test_set_widths_equals_a_model_trained_with_that_level():
    # An untrained width through set_widths == the same width built as a real level.
    m, x = _model(_cfg()), _x()
    cfg9 = _cfg(fractions=(0.25, 0.5, 0.75, 1.0))  # 0.25 was never a level of m
    m9 = cfg9.make_model().eval()
    missing, unexpected = m9.load_state_dict(_no_rope(m.state_dict()), strict=False)
    assert all(".cos_" in k or ".sin_" in k for k in missing) and not unexpected
    m9.set_level_use(0)
    m.set_widths(cfg9.hidden_dims[0], cfg9.intermediate_dims[0], cfg9.num_heads[0])
    torch.testing.assert_close(m(x), m9(x))


def test_set_widths_per_layer_mix_runs_and_rejects_bad_heads():
    cfg = _cfg()
    m, x = _model(cfg), _x()
    m.set_widths(128, [192, 384, 288], [4, 8, 6])
    assert m(x).shape == (2, 12, V)
    with pytest.raises(AssertionError):
        m.set_widths(128, 192, 5)  # not a multiple of the GQA ratio 2


def test_level_norms_start_as_identity_and_interpolate():
    plain, x = _model(_cfg()), _x()
    cfg = _cfg(level_norms=True)
    m = cfg.make_model().eval()
    m.load_state_dict(plain.state_dict(), strict=False)
    for lvl in range(3):
        plain.set_level_use(lvl), m.set_level_use(lvl)
        torch.testing.assert_close(m(x), plain(x))
    norm = m.layers[0].input_layernorm
    with torch.no_grad():
        norm.level_delta.normal_()
    mid = (cfg.hidden_dims[0] + cfg.hidden_dims[1]) // 2  # 160, between 128 and 192
    norm.dim_override = mid
    want = norm.weight[:mid] + 0.5 * (norm.level_delta[0, :mid] + norm.level_delta[1, :mid])
    torch.testing.assert_close(norm.effective_weight(), want)
    norm.dim_override = None
    norm.set_level_use(1)
    torch.testing.assert_close(norm.effective_weight(), norm.weight[:192] + norm.level_delta[1, :192])


@pytest.mark.parametrize("mode", ["scale", "scale_shift"])
def test_modulation_identity_at_start_and_at_full_width(mode):
    plain, x = _model(_cfg()), _x()
    m = _cfg(modulation=mode).make_model().eval()
    m.load_state_dict(plain.state_dict(), strict=False)
    m.set_level_use(0), plain.set_level_use(0)
    torch.testing.assert_close(m(x), plain(x))  # zero-initialised last layer
    with torch.no_grad():
        for b in m.layers:
            b.attn_mod.net[-1].weight.normal_(0, 0.1)
            b.mlp_mod.net[-1].weight.normal_(0, 0.1)
    assert (m(x) - plain(x)).abs().max() > 1e-4  # sub-widths are modulated
    m.set_level_use(2), plain.set_level_use(2)
    torch.testing.assert_close(m(x), plain(x))  # full width never is


def test_modulation_scale_folds_into_export_and_shift_is_refused(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from training.export_flex_hf import flexllama_level_state_dict, hf_level_config
    hf_cfg = Qwen3Config(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=L,
                         num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD,
                         max_position_embeddings=32, tie_word_embeddings=True, rope_theta=10000.0,
                         rms_norm_eps=1e-6)
    cfg = _cfg(modulation="scale", level_norms=True)
    m = _model(cfg)
    with torch.no_grad():
        for b in m.layers:
            b.attn_mod.net[-1].weight.normal_(0, 0.1)
            b.mlp_mod.net[-1].weight.normal_(0, 0.1)
            b.input_layernorm.level_delta.normal_(0, 0.1)
    x = _x()
    m.set_level_use(0)
    ref = m(x)
    hf = Qwen3ForCausalLM(hf_level_config(hf_cfg, cfg, 0)).eval()
    hf.load_state_dict(flexllama_level_state_dict(m, 0), strict=True)
    torch.testing.assert_close(hf(input_ids=x).logits, ref, atol=2e-5, rtol=2e-5)
    ms = _model(_cfg(modulation="scale_shift"))
    with torch.no_grad():
        ms.layers[0].mlp_mod.net[-1].bias.fill_(0.1)
    with pytest.raises(NotImplementedError):
        flexllama_level_state_dict(ms, 0)


def test_anchor_sampling_always_keeps_anchors():
    from training.trainer import FlexLMTrainer
    fake = types.SimpleNamespace(
        submodel=types.SimpleNamespace(max_level=lambda: 8),
        training_context=types.SimpleNamespace(level_sampling="sandwich", num_levels_per_step=4,
                                               anchor_levels=(0, 4, 8), seed=0),
        global_step=0)
    seen = set()
    for step in range(50):
        fake.global_step = step
        lv = FlexLMTrainer._select_levels(fake, "train")
        assert len(lv) == 4 and {0, 4, 8} <= set(lv)
        seen |= set(lv) - {0, 4, 8}
    assert seen == {1, 2, 3, 5, 6, 7}
    assert FlexLMTrainer._select_levels(fake, "val") == list(range(9))


def test_continuum_params_get_their_own_lr():
    from training.trainer import adamw
    m = _cfg(level_norms=True, modulation="scale").make_model()
    opt = adamw(m.parameters(), lr=1e-4, weight_decay=0.1, model=m, continuum_lr=1e-3)
    assert len(opt.param_groups) == 2 and opt.param_groups[1]["lr"] == 1e-3
    assert opt.param_groups[1]["weight_decay"] == 0.0
    n_extra = sum(p.numel() for p in opt.param_groups[1]["params"])
    want = sum(p.numel() for n, p in m.named_parameters() if "level_delta" in n or "_mod." in n)
    assert n_extra == want > 0
    plain = _cfg().make_model()
    assert len(adamw(plain.parameters(), 1e-4, 0.0, model=plain, continuum_lr=1e-3).param_groups) == 1


def test_eval_helpers_dims_and_param_counts():
    from eval.eval_continuum import count_params, mixnmatch_candidates, uniform_dims
    cfg = _cfg()
    for lvl, f in enumerate((0.5, 0.75, 1.0)):
        assert uniform_dims(cfg, f) == dict(hidden=cfg.hidden_dims[lvl], intermediate=cfg.intermediate_dims[lvl],
                                            heads=cfg.num_heads[lvl])
    m = _model(cfg)
    m.set_level_use(2)
    n_real = sum(p.numel() for n, p in m.named_parameters())
    tot, body = count_params(cfg, H, [I] * L, [NH] * L)
    assert tot == n_real and tot - body == V * H
    c = mixnmatch_candidates(cfg, [0.5, 0.75, 1.0], ks=(1, 2))
    assert len(c) == 2 * 2 * 2 * 3
    lo, hi = count_params(cfg, 128, [192] * L, [4] * L)[0], count_params(cfg, 192, [288] * L, [6] * L)[0]
    for x in c[:12]:
        assert lo < count_params(cfg, x["hidden"], x["intermediate"], x["heads"])[0] < hi


def test_stage1_configs_resolve():
    from training.run_experiment import resolve_from_str
    from training.stage1 import CORRECTED_TEACHER_1P7B
    b = resolve_from_str("flexllama,s1.qwen3_0p6b.pilot.kd_corrected")
    assert b.training_context.teacher_hf_model == CORRECTED_TEACHER_1P7B
    t = resolve_from_str("flexllama,s1.qwen3_1p7b.teacher_correction")
    assert t.training_context.kd_lambda == 0 and t.training_context.gradient_clip_val is None
    assert list(t.model_config.hidden_dims) == [2048]
    assert resolve_from_str("flexllama,s1.qwen3_0p6b.pilot.mod_scale").model_config.modulation == "scale"
    assert resolve_from_str("flexllama,s1.qwen3_0p6b.pilot.level_norms").model_config.level_norms
    g = resolve_from_str("flexllama,s1.qwen3_0p6b.pilot.grid9_anchors_plus1").training_context
    assert g.anchor_levels == (0, 4, 8) and g.num_levels_per_step == 4
    w = resolve_from_str("flexllama,s1.qwen3_0p6b.full.w025").model_config
    assert list(w.hidden_dims) == [256, 512, 768, 1024] and list(w.num_heads) == [4, 8, 12, 16]


def test_memopt_kd_trainer_with_kd0_needs_no_teacher_and_equals_ce():
    # Pilot kd0 and the teacher correction run the KD trainer with kd_lambda 0: the teacher
    # is never loaded, so the step must not call it, and the gradient must equal plain CE.
    import types as _t
    from config.experiments import LLaMAKDTrainingContext, LLaMATrainingContext
    from training.memopt import MemOptFlexLMKDTrainer, MemOptFlexLMTrainer
    cfg = FlexLLaMAConfig(vocab_size=V, max_seq_length=32, num_layers=2, hidden_dims=(32, 64),
                          num_heads=(4, 8), num_kv_heads=(4, 8), intermediate_dims=(64, 128))

    def stub(m):
        m.log = lambda *a, **k: None
        m.optimizers = lambda: _t.SimpleNamespace(zero_grad=lambda: m.zero_grad(set_to_none=True), step=lambda: None)
        m.manual_backward = lambda loss, **kw: loss.backward(**kw)
        m.train()
        return m

    torch.manual_seed(0)
    a = stub(MemOptFlexLMTrainer(cfg, LLaMATrainingContext(epochs=1, wandb_project_name=None)))
    ctx = LLaMAKDTrainingContext(kd_lambda=0.0, kd_temperature=2.0, epochs=1, wandb_project_name=None)
    b = stub(MemOptFlexLMKDTrainer(cfg, ctx))
    b.load_state_dict(a.state_dict())
    assert b._teacher is None
    ids = torch.randint(0, V, (4, 16), generator=torch.Generator().manual_seed(2))
    for m in (a, b):
        m._optimizer_step = lambda opt: None
        m._step((ids, ids), "train")
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pb.grad, pa.grad, msg=n)
