"""
Saves the model weights from a Lightning checkpoint to saved_models/<config>.pt,
the file utils.save_model() would have written at the end of training, and
optionally exports every level as a Hugging Face model (export_flex_hf.py).

For runs whose training finished but the post-fit reload in training.finetune
failed: there every rank loaded best-model.ckpt onto the GPU it was saved from
(cuda:0), which runs out of memory at Qwen scale. Here the checkpoint is
memory-mapped on the CPU and only the model weights are read.

    python recover_ckpt.py flexllama,sft.medmix_qwen3_1p7b --export
"""
import argparse
import os

import torch


def submodel_state_dict(ckpt: dict) -> dict:
    prefix = "submodel."
    return {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--ckpt", default=None, help="default: checkpoints/<config>/best-model.ckpt")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    import config.paths as paths
    import utils
    from run_experiment import resolve_from_str

    safe = utils.make_str_filename_safe(args.config)
    path = args.ckpt or os.path.join(paths.CHECKPOINT_PATH, safe, "best-model.ckpt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    print(f"{path}: epoch {ckpt.get('epoch')}, global_step {ckpt.get('global_step')}")

    cfg = resolve_from_str(args.config).model_config
    model = cfg.make_model()
    model.load_state_dict(submodel_state_dict(ckpt))
    print("weights load: ok (all keys match)")
    utils.save_model(args.config, model)
    print(f"saved {paths.TRAINED_MODELS / (safe + '.pt')}")

    if args.export:
        import export_flex_hf
        model.eval()
        export_flex_hf.export(model, cfg, cfg.pretrained_hf_model, os.path.join("hf_export", safe))


if __name__ == "__main__":
    main()
