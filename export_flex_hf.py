"""
Exports every level of a FlexLLaMA as a standalone Hugging Face model, so each
level can be evaluated with standard tools: lm-eval through evalhf.sh (cloze / MCF)
and HF generate with a KV cache (eval_gen.py, FaithMed's generated-answer format).

A level is a slice of the max-level weights with the same architecture, so it is
an ordinary checkpoint of the source model family (llama / qwen2 / qwen3) at a
smaller width: hidden_dims[l], num_heads[l], num_kv_heads[l], intermediate_dims[l],
same head_dim, layers, vocab and RoPE.

    python export_flex_hf.py flexllama,sft.medmix_qwen3_1p7b             # saved_models/<config>.pt
    python export_flex_hf.py flexllama,sft.medmix_qwen3_1p7b --init hf   # untrained: the HF warm start
writes hf_export/<config>[.init]/level<i>/ (config, bf16 weights, tokenizer).
"""
import argparse
import os

import torch


def flexllama_level_state_dict(model, level: int) -> dict:
    """HF-layout state dict of `model` at `level` (model is left at that level)."""
    model.set_level_use(level)
    sd = {"model.embed_tokens.weight": model.embed_tokens.make_base_copy().weight,
          "model.norm.weight": model.norm.make_base_copy().weight}
    for i, block in enumerate(model.layers):
        pfx = f"model.layers.{i}"
        attn = block.self_attn
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, proj).make_base_copy()
            sd[f"{pfx}.self_attn.{proj}.weight"] = lin.weight
            if lin.bias is not None:
                sd[f"{pfx}.self_attn.{proj}.bias"] = lin.bias
        if getattr(attn, "qk_norm", False):
            sd[f"{pfx}.self_attn.q_norm.weight"] = attn.q_norm.weight
            sd[f"{pfx}.self_attn.k_norm.weight"] = attn.k_norm.weight
        sd[f"{pfx}.input_layernorm.weight"] = block.input_layernorm.make_base_copy().weight
        sd[f"{pfx}.post_attention_layernorm.weight"] = block.post_attention_layernorm.make_base_copy().weight
        for proj in ("gate_proj", "up_proj", "down_proj"):
            sd[f"{pfx}.mlp.{proj}.weight"] = getattr(block.mlp, proj).make_base_copy().weight
    if model.config.tie_embeddings:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    else:
        sd["lm_head.weight"] = model.lm_head.current_layer().weight
    return {k: v.detach().clone() for k, v in sd.items()}


def hf_level_config(source_hf_config, cfg, level: int):
    """The source model's HF config with the widths of `level`."""
    c = source_hf_config.__class__.from_dict(source_hf_config.to_dict())
    hidden, heads = list(cfg.hidden_dims)[level], list(cfg.num_heads)[level]
    c.hidden_size = hidden
    c.num_attention_heads = heads
    c.num_key_value_heads = list(cfg.num_kv_heads)[level]
    c.intermediate_size = list(cfg.intermediate_dims)[level]
    if hasattr(c, "head_dim"):
        c.head_dim = hidden // heads
    return c


def export(model, cfg, source: str, out_dir: str, levels=None, dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    source_config = AutoConfig.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    paths = []
    for level in levels if levels is not None else range(model.max_level() + 1):
        hf = AutoModelForCausalLM.from_config(hf_level_config(source_config, cfg, level))
        hf.load_state_dict(flexllama_level_state_dict(model, level), strict=True)
        path = os.path.join(out_dir, f"level{level}")
        hf.to(dtype).save_pretrained(path)
        tokenizer.save_pretrained(path)
        n = sum(p.numel() for p in hf.parameters())
        print(f"level {level}: {n / 1e6:.0f}M parameters -> {path}")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--init", choices=["trained", "hf"], default="trained",
                        help="trained: saved_models/<config>.pt; hf: the untrained Hugging Face warm start")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import utils
    from run_experiment import resolve_from_str
    cfg = resolve_from_str(args.config).model_config
    if args.init == "hf":
        model = cfg.make_model()
        utils.load_llama_weights_into_flexllama(model, cfg.pretrained_hf_model)
    else:
        model = utils.load_model(args.config, cfg)
    model.eval()
    safe = utils.make_str_filename_safe(args.config) + (".init" if args.init == "hf" else "")
    export(model, cfg, cfg.pretrained_hf_model, args.out or os.path.join("hf_export", safe))


if __name__ == "__main__":
    main()
