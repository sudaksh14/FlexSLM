"""M1 importance ordering (training/importance_order.py) on tiny random Qwen3 models."""
import pytest
import torch

from training.importance_order import (apply_permutation, collect_taylor_importance,
                                       permutation_from_importance, permuted_logit_diff)


def _qwen3(tie: bool, hidden=32, inter=48, heads=4, kv=2):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=2,
                      num_attention_heads=heads, num_key_value_heads=kv, head_dim=16,  # decoupled head_dim
                      max_position_embeddings=64, tie_word_embeddings=tie)
    model = Qwen3ForCausalLM(cfg).float().eval()
    # Random norm weights, so a missed RMSNorm permutation would show.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
    return model


def _batches(n=2):
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 97, (2, 12), generator=g) for _ in range(n)]


@pytest.mark.parametrize("tie", [True, False])
def test_random_permutation_preserves_the_function(tie):
    model = _qwen3(tie)
    g = torch.Generator().manual_seed(3)
    imp = dict(mlp=[torch.rand(48, generator=g) for _ in range(2)],
               groups=[torch.rand(2, generator=g) for _ in range(2)],
               hidden=torch.rand(32, generator=g))
    perm = permutation_from_importance(imp)
    assert not torch.equal(perm["hidden"], torch.arange(32))
    diff, scale = permuted_logit_diff(model, perm, _batches(1)[0])
    assert diff <= 1e-5 + 1e-5 * scale


def test_wrong_permutation_is_detected():
    model = _qwen3(True)
    perm = permutation_from_importance(dict(
        mlp=[torch.arange(48.0)] * 2, groups=[torch.arange(2.0)] * 2, hidden=torch.arange(32.0)))
    # Permute the residual stream but not the final norm: the function must change.
    model.model.norm.weight.data = model.model.norm.weight.data.flip(0).clone()
    before = model(input_ids=_batches(1)[0]).logits
    apply_permutation(model, perm)
    model.model.norm.weight.data = model.model.norm.weight.data.flip(0).clone()
    after = model(input_ids=_batches(1)[0]).logits
    assert (before - after).abs().max() > 1e-3


def test_taylor_importance_shapes_hooks_and_no_param_grads():
    model = _qwen3(True)
    imp = collect_taylor_importance(model, _batches())
    assert [s.shape for s in imp["mlp"]] == [torch.Size([48])] * 2
    assert [s.shape for s in imp["groups"]] == [torch.Size([2])] * 2
    assert imp["hidden"].shape == torch.Size([32])
    assert imp["tokens"] == 2 * 2 * 12
    assert all(s.min() >= 0 and s.sum() > 0 for s in imp["mlp"] + imp["groups"] + [imp["hidden"]])
    assert all(p.grad is None for p in model.parameters())
    for m in model.modules():
        assert not m._forward_pre_hooks


def test_ordering_puts_important_units_first():
    # The ordered model's importance is sorted (most important first) everywhere.
    model = _qwen3(False)
    perm = permutation_from_importance(collect_taylor_importance(model, _batches()))
    apply_permutation(model, perm)
    imp = collect_taylor_importance(model, _batches())
    for s in imp["mlp"] + imp["groups"] + [imp["hidden"]]:
        assert torch.all(s[:-1] >= s[1:] * (1 - 1e-4) - 1e-9)


def test_ordered_checkpoint_warm_starts_flexllama(tmp_path):
    # Save an ordered tiny Qwen3, warm-start a 3-level FlexLLaMA from it: the max level
    # reproduces the original (unpermuted) model and every level runs.
    from config.experiments import _flex_llama_dims_fixed_head
    from networks.flexllama import FlexLLaMAConfig
    from utils import load_llama_weights_into_flexllama
    model = _qwen3(True, hidden=256, inter=384, heads=8, kv=4)
    x = _batches(1)[0]
    with torch.no_grad():
        ref = model(input_ids=x).logits
    apply_permutation(model, permutation_from_importance(collect_taylor_importance(model, _batches())))
    model.save_pretrained(tmp_path)
    hd, hs, kvs, im = _flex_llama_dims_fixed_head(256, 8, 4, 384, fractions=(0.5, 0.75, 1.0), head_dim=16)
    cfg = FlexLLaMAConfig(vocab_size=97, max_seq_length=64, num_layers=2, hidden_dims=hd, num_heads=hs,
                          num_kv_heads=kvs, intermediate_dims=im, rope_theta=model.config.rope_theta,
                          rms_norm_eps=model.config.rms_norm_eps, tie_embeddings=True, qkv_bias=False,
                          qk_norm=True, pretrained_hf_model=str(tmp_path), head_dim=16)
    flex = cfg.make_model().eval()
    load_llama_weights_into_flexllama(flex, str(tmp_path))
    with torch.no_grad():
        flex.set_level_use(flex.max_level())
        torch.testing.assert_close(flex(x), ref, atol=1e-4, rtol=1e-4)
        flex.set_level_use(0)
        assert torch.isfinite(flex(x)).all()


def test_hidden_activation_score_and_start_weights():
    from training.importance_order import ORDERED_CHECKPOINTS, collect_hidden_activation, start_weights
    model = _qwen3(True)
    h = collect_hidden_activation(model, _batches())
    assert h.shape == torch.Size([32]) and h.min() >= 0 and h.sum() > 0
    assert all(not m._forward_pre_hooks for m in model.modules())
    assert start_weights("none") == "Qwen/Qwen3-0.6B"
    assert start_weights("hidden_act") == ORDERED_CHECKPOINTS["hidden_act"]
    with pytest.raises(ValueError):
        start_weights("random")
