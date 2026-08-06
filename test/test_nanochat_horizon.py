"""
Prints the nanochat-recipe token horizon for each teacher preset, and sanity
checks flexllama_scaling_params against a real built model.
"""
import torch

from config.experiments import (TEACHER_PRESETS, _flex_llama_dims,
                                flexllama_scaling_params, nanochat_token_horizon)
from networks.flexllama import FlexLLaMAConfig

SAMPLE_10BT_TOKENS = 10_000_000_000


def config_for(name, n_levels=3):
    p = TEACHER_PRESETS[name]
    hd, nh, kv, im = _flex_llama_dims(**p["dims_spec"], n_levels=n_levels)
    return FlexLLaMAConfig(
        vocab_size=p["vocab_size"], num_layers=p["num_layers"],
        hidden_dims=hd, num_heads=nh, num_kv_heads=kv, intermediate_dims=im,
        rope_theta=p["rope_theta"], rms_norm_eps=p["rms_norm_eps"],
        tie_embeddings=p["tie_embeddings"], qkv_bias=p["qkv_bias"],
    )


print("=== analytic scaling-param count vs. a really built model ===")
small = FlexLLaMAConfig(
    vocab_size=1000, num_layers=3, hidden_dims=(32, 64, 96),
    num_heads=(4, 8, 12), num_kv_heads=(4, 8, 12),
    intermediate_dims=(64, 128, 192), tie_embeddings=False, qkv_bias=False)
model = small.make_model()
built = sum(p.numel() for p in model.layers.parameters())
built += model.lm_head.level2.weight.numel()  # max-level lm_head
analytic = flexllama_scaling_params(small)
print(f"  built={built:,}  analytic={analytic:,}  diff={built - analytic:,}")
assert built == analytic, "analytic formula disagrees with the built model"
print("  analytic formula matches the real model exactly: OK")

print("\n=== nanochat horizons per teacher (ratio 8 = speedrun, 12 = their default) ===")
print(f"{'teacher':<16} {'scaling params':>15} {'ratio 8':>14} {'ratio 12':>14}  fits 10BT?")
print("-" * 78)
rows = []
for name in ("llama-160m", "smollm2-360m", "tinyllama-1.1b", "qwen2.5-0.5b",
             "qwen2.5-1.5b", "smollm2-1.7b"):
    cfg = config_for(name)
    sp = flexllama_scaling_params(cfg)
    h8 = nanochat_token_horizon(cfg, 8)
    h12 = nanochat_token_horizon(cfg, 12)
    fits = "yes" if h12 <= SAMPLE_10BT_TOKENS else ("ratio 8 only" if h8 <= SAMPLE_10BT_TOKENS else "NO")
    print(f"{name:<16} {sp:>15,} {h8:>14,} {h12:>14,}  {fits}")
    rows.append((name, sp, h8, h12))

print("\n=== disk cost of preparing at ratio 12 (2 bytes/token, 4 if vocab > 65535) ===")
for name, sp, h8, h12 in rows:
    cfg = config_for(name)
    width = 2 if cfg.vocab_size < 2 ** 16 else 4
    print(f"  {name:<16} {h12/1e9:>6.2f}B tokens -> {h12*width/2**30:>7.1f} GiB "
          f"(uint{width*8})")

print("\n=== prepare commands for the nanochat recipe ===")
for name, sp, h8, h12 in rows:
    hf = TEACHER_PRESETS[name]["hf_model"]
    budget = min(h12, SAMPLE_10BT_TOKENS)
    print(f"  sbatch prepare_fineweb.sh {hf} {budget:.3e}")

print("\nDONE")
