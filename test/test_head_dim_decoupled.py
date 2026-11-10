"""
CPU test for FlexLLaMAConfig.head_dim (Qwen3-0.6B-style checkpoints whose
head_dim != hidden // heads) and the GQA-preserving level builder.

Builds tiny random Qwen3 checkpoints (no download), warm-starts a FlexLLaMA from
them and checks: (1) the max level reproduces the HF logits, (2) every exported
level reproduces the FlexLLaMA logits at that level, (3) the original convention
(head_dim None, used by the existing Qwen3-1.7B runs) is unchanged.

    python -m pytest test/test_head_dim_decoupled.py -q
"""
import os
import tempfile

import torch

from config.experiments import _flex_llama_dims, _flex_llama_dims_fixed_head
from training.export_flex_hf import export
from networks.flexllama import FlexLLaMAConfig
from utils import load_llama_weights_into_flexllama


def _tiny_qwen3(path, hidden, heads, kv, head_dim, inter):
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=hidden, num_attention_heads=heads,
                      num_key_value_heads=kv, head_dim=head_dim, intermediate_size=inter,
                      num_hidden_layers=2, tie_word_embeddings=True, rope_theta=1e6,
                      rms_norm_eps=1e-6, max_position_embeddings=64)
    model = Qwen3ForCausalLM(cfg).eval()
    model.save_pretrained(path)
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    tok.save_pretrained(path)
    return model


def _run(hidden, heads, kv, head_dim, inter, fractions, flex_head_dim):
    from transformers import AutoModelForCausalLM
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        hf = _tiny_qwen3(src, hidden, heads, kv, head_dim, inter)
        dims = _flex_llama_dims(hidden, heads, kv, inter, fractions=fractions, head_dim=flex_head_dim)
        hd, hs, kvs, im = dims
        cfg = FlexLLaMAConfig(vocab_size=97, max_seq_length=64, num_layers=2, hidden_dims=hd,
                              num_heads=hs, num_kv_heads=kvs, intermediate_dims=im,
                              rope_theta=1e6, rms_norm_eps=1e-6, tie_embeddings=True,
                              qkv_bias=False, qk_norm=True, pretrained_hf_model=src,
                              head_dim=flex_head_dim)
        model = cfg.make_model().eval()
        load_llama_weights_into_flexllama(model, src)
        x = torch.randint(0, 97, (2, 11))
        with torch.no_grad():
            model.set_level_use(model.max_level())
            assert torch.allclose(model(x), hf(x).logits, atol=1e-4), "max level != HF"
            paths = export(model, cfg, src, os.path.join(d, "out"), dtype=torch.float32)
            for level, p in enumerate(paths):
                model.set_level_use(level)
                ref = model(x)
                got = AutoModelForCausalLM.from_pretrained(p).eval()(x).logits
                assert torch.allclose(ref, got, atol=1e-4), f"exported level {level} != FlexLLaMA"
        return dims


def test_fixed_head_dims_keep_gqa_ratio():
    hd, hs, kvs, im = _flex_llama_dims_fixed_head(1024, 16, 8, 3072, 128, [0.5, 0.75, 1.0])
    assert (hd, hs, kvs, im) == ((512, 768, 1024), (8, 12, 16), (4, 6, 8), (1536, 2304, 3072))
    assert all(h // k == 2 for h, k in zip(hs, kvs))


def test_decoupled_head_dim_qwen3_06b_style():
    # hidden 128 with 4 heads of size 64: q dim 256 != hidden, like Qwen3-0.6B.
    dims = _run(hidden=128, heads=4, kv=2, head_dim=64, inter=256, fractions=[0.5, 1.0], flex_head_dim=64)
    assert dims == ((64, 128), (2, 4), (1, 2), (128, 256))


def test_original_convention_unchanged():
    # head_dim == hidden // heads and head_dim=None in the FlexLLaMA config (the
    # Qwen3-1.7B path used by the existing runs).
    _run(hidden=128, heads=4, kv=2, head_dim=32, inter=256, fractions=[0.5, 1.0], flex_head_dim=None)


def test_dense_grid_with_repeated_head_counts():
    # 4 levels over 8 heads / 4 KV groups: head counts repeat (4, 4, 6, 8) while
    # hidden and intermediate widths grow; max level must still match HF.
    dims = _run(hidden=512, heads=8, kv=4, head_dim=32, inter=1024,
                fractions=(0.5, 0.625, 0.75, 1.0), flex_head_dim=32)
    assert dims[1][0] == dims[1][1]
