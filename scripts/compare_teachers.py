"""
Root-cause analysis of the weak T1 teacher (implementation_log.md, 2026-11-11).
CPU job. Loads the Hugging Face base Qwen3-1.7B, the student's old fine-tuned teacher
(old_ckpts/saved_models/..._qwen3_1p7b_teacher.pt, avg 37.9 in the student's report) and our T1
teacher (store/saved_models/flexllama__sft.paper.qwen3_1p7b_teacher.pt, avg 30.6), all as
FlexLLaMA state dicts, and reports per tensor group:
  rel = ||W_ft - W_base|| / ||W_base||     (how far each fine-tune moved the weights)
  cos = cosine(old delta, new delta)       (did both move in the same direction)
then exports the old teacher as a Hugging Face model (hf_export/old_teacher_1p7b/level2) so that
eval/eval_gen.py can score it with the same pipeline as ours.
    python scripts/compare_teachers.py [--old PATH] [--new PATH]
"""
import argparse
import collections
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
OLD = "/var/scratch/skalra/flexslm/old_ckpts/saved_models/flexxxxllamaxnsftxmmedmixxxx_qwen3_1p7b_teacher.pt"
NEW = os.path.join(os.environ.get("FLEXSLM_STORE", "/var/scratch/skalra/flexslm/store"),
                   "saved_models", "flexllama__sft.paper.qwen3_1p7b_teacher.pt")


def group(name: str) -> str:
    for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "q_norm", "k_norm", "input_layernorm", "post_attention_layernorm"):
        if key in name:
            return key
    return "embed/lm_head" if ("embed" in name or "lm_head" in name) else ("final_norm" if "norm" in name else "other")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", default=OLD)
    p.add_argument("--new", default=NEW)
    p.add_argument("--no-export", action="store_true")
    a = p.parse_args()

    import utils
    from training.run_experiment import resolve_from_str
    cfg = resolve_from_str("flexllama,sft.paper.qwen3_1p7b_teacher").model_config
    base = cfg.make_model()
    utils.load_llama_weights_into_flexllama(base, cfg.pretrained_hf_model)
    base_sd = {k: v.float() for k, v in base.state_dict().items()}
    old_sd = torch.load(a.old, map_location="cpu")
    new_sd = torch.load(a.new, map_location="cpu")
    print(f"tensors: base {len(base_sd)} old {len(old_sd)} new {len(new_sd)}; "
          f"missing in old {len(set(base_sd) - set(old_sd))} extra in old {len(set(old_sd) - set(base_sd))}; "
          f"missing in new {len(set(base_sd) - set(new_sd))}", flush=True)

    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for k, b in base_sd.items():
        if k not in old_sd or k not in new_sd or old_sd[k].shape != b.shape or new_sd[k].shape != b.shape:
            print("  skipped (missing or shape mismatch):", k, flush=True)
            continue
        do, dn = old_sd[k].float() - b, new_sd[k].float() - b
        g = acc[group(k)]
        g["n"] += b.numel()
        g["base2"] += float((b * b).sum())
        g["old2"] += float((do * do).sum())
        g["new2"] += float((dn * dn).sum())
        g["dot"] += float((do * dn).sum())
        g["old_abs"] += float(do.abs().sum())
        g["new_abs"] += float(dn.abs().sum())
    tot = collections.defaultdict(float)
    print(f"\n{'group':26s} {'params':>10s} {'rel_old':>9s} {'rel_new':>9s} {'new/old':>8s} {'cos':>7s}")
    for name, g in sorted(acc.items(), key=lambda kv: -kv[1]["n"]):
        for k, v in g.items():
            tot[k] += v
        print(f"{name:26s} {g['n'] / 1e6:9.1f}M {math.sqrt(g['old2'] / g['base2']):9.5f} "
              f"{math.sqrt(g['new2'] / g['base2']):9.5f} {math.sqrt(g['new2'] / max(g['old2'], 1e-30)):8.2f} "
              f"{g['dot'] / math.sqrt(max(g['old2'] * g['new2'], 1e-30)):7.3f}")
    g = tot
    print(f"{'ALL':26s} {g['n'] / 1e6:9.1f}M {math.sqrt(g['old2'] / g['base2']):9.5f} "
          f"{math.sqrt(g['new2'] / g['base2']):9.5f} {math.sqrt(g['new2'] / g['old2']):8.2f} "
          f"{g['dot'] / math.sqrt(g['old2'] * g['new2']):7.3f}")
    print(f"mean |delta| per weight: old {g['old_abs'] / g['n']:.3e}  new {g['new_abs'] / g['n']:.3e}", flush=True)

    if not a.no_export:
        from training.export_flex_hf import export
        model = cfg.make_model()
        model.load_state_dict(old_sd, strict=True)
        export(model, cfg, cfg.pretrained_hf_model, os.path.join("hf_export", "old_teacher_1p7b"), levels=[2])


if __name__ == "__main__":
    main()
