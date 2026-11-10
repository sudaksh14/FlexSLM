"""
Checks the Qwen3 warm start on the real checkpoint: the max level of a
FlexLLaMA config (default flexllama,sft.medmix_qwen3_1p7b), loaded through
utils.load_llama_weights_into_flexllama, must give the same logits as
Hugging Face's own Qwen3-1.7B. Then exports every level of that untrained
model (export_flex_hf.py) as the pre-SFT baseline.

    python -m scripts.check_qwen3_warmstart [config] [--no-export]
    sbatch scripts/qwen3_check.sh        # 1 GPU, default config
"""
import sys

import torch


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import utils
    import training.export_flex_hf as export_flex_hf
    from training.run_experiment import resolve_from_str

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    export = "--no-export" not in sys.argv
    conf = args[0] if args else "flexllama,sft.medmix_qwen3_1p7b"
    cfg = resolve_from_str(conf).model_config
    name = cfg.pretrained_hf_model
    print(f"config {conf}: levels hidden {list(cfg.hidden_dims)} heads {list(cfg.num_heads)} "
          f"kv {list(cfg.num_kv_heads)} mlp {list(cfg.intermediate_dims)} qk_norm {cfg.qk_norm} "
          f"head_dim {getattr(cfg, 'head_dim', None)}")

    flex = cfg.make_model()
    utils.load_llama_weights_into_flexllama(flex, name)
    flex.eval().cuda()
    hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval().cuda()
    tok = AutoTokenizer.from_pretrained(name)

    texts = ["Question: A 45-year-old man presents with crushing chest pain radiating to the left arm. "
             "Which enzyme rises first?\nAnswer:",
             "The first-line treatment for uncomplicated hypertension in adults is"]
    worst = 0.0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.cuda()
            ref, out = hf(ids).logits, flex(ids)
            diff = (ref - out).abs().max().item()
            same_top = (ref.argmax(-1) == out.argmax(-1)).float().mean().item()
            worst = max(worst, diff)
            print(f"max |logit diff| {diff:.2e}, same argmax {100 * same_top:.0f}%  ({ids.shape[1]} tokens)")
    del hf
    ok = worst < 1e-2
    print("WARM START MATCHES HF" if ok else "MISMATCH: do not train")
    if not ok:
        sys.exit(1)

    if not export:
        return
    flex.cpu()
    safe = utils.make_str_filename_safe(conf) + ".init"
    export_flex_hf.export(flex, cfg, name, f"hf_export/{safe}")


if __name__ == "__main__":
    main()
