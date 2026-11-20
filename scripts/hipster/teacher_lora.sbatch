#!/usr/bin/env bash
#SBATCH --job-name=flexslm_teacher_lora
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00:00
#SBATCH --output=jobs/%x-%j.out
# LoRA fine-tune of HuatuoGPT-o1-7B on medmix (answer-only) = the Stage-2 KD teacher candidate
# (training/finetune_teacher_lora.py). One task, torchrun starts one process per allocated GPU.
# performance: RTX 6000 Ada 48 GB, 32 CPUs per GPU enforced (4 GPUs -> 128 CPUs). No --exclusive, no --mem.
# --time = 2 x a pessimistic 5 h; the smoke test prints the measured projection, adjust --time from it.
#
# 1) SMOKE FIRST (2 GPUs, 20 steps of the LONGEST sequences, merge + save + verify into scratch, then deleted):
#      sbatch --gres=gpu:rtx_6000_ada:2 --cpus-per-task=64 --time=01:30:00 --export=ALL,SMOKE_STEPS=20 \
#             scripts/hipster/teacher_lora.sbatch
# 2) Full run (4 GPUs, global batch 32 = 4 x 8, ~1,043 steps):
#      sbatch [--time=HH:MM:SS] scripts/hipster/teacher_lora.sbatch
# Extra arguments of the python script can be passed after the sbatch script, e.g. `... teacher_lora.sbatch --lr 2e-4`.
# Result: hf_export/medmix_huatuogpt_o1_7b_lora (merged HF dir, vocab trimmed to 151,936 = Qwen3's, Qwen3 tokenizer),
# adapters in $FLEXSLM_STORE/teacher_lora/<job>/ (final adapter also copied to ~/flexslm_saves/teacher_lora/<job>).
source scripts/hipster/common.sh gpu
BASE=${BASE:-FreedomIntelligence/HuatuoGPT-o1-7B}
EXTRA=("$@")
NPROC=$(python -c "import torch; print(torch.cuda.device_count())")

# Home is small and shared (rules: check before adding big things). The 7B bf16 weights are ~15 GB in the HF cache.
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "[teacher_lora] home free ${FREE_GB} GB, HF_HOME=$HF_HOME"
# One process downloads (or finds in the cache) the base model, so the ranks do not race on the same files.
python - <<EOF
import os, sys
from huggingface_hub import snapshot_download, try_to_load_from_cache
cached = try_to_load_from_cache("$BASE", "config.json")
if not isinstance(cached, str) and int("$FREE_GB") < 30:
    sys.exit("[teacher_lora] not cached and < 30 GB free in home: free space first (see hipster rules)")
print("[teacher_lora] base model at", snapshot_download("$BASE", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
EOF

if [[ -n "${SMOKE_STEPS:-}" ]]; then
  export FLEXSLM_STORE=/scratch/$USER/flexslm/smoke_store/$SLURM_JOB_ID
  export WANDB_MODE=disabled
  MERGE_DIR=$FLEXSLM_STORE/merged
  echo "[teacher_lora] SMOKE ${SMOKE_STEPS} steps on ${NPROC} GPU(s), store=$FLEXSLM_STORE"
  torchrun --standalone --nproc_per_node="$NPROC" -m training.finetune_teacher_lora \
    --base "$BASE" --smoke_steps "$SMOKE_STEPS" --smoke_merge_dir "$MERGE_DIR" "${EXTRA[@]}"
  rm -rf "$FLEXSLM_STORE"
else
  echo "[teacher_lora] FULL run on ${NPROC} GPU(s), store=$FLEXSLM_STORE"
  torchrun --standalone --nproc_per_node="$NPROC" -m training.finetune_teacher_lora --base "$BASE" "${EXTRA[@]}"
  mkdir -p "$HOME/flexslm_saves/teacher_lora"
  cp -r "$FLEXSLM_STORE/teacher_lora/$SLURM_JOB_ID/final" "$HOME/flexslm_saves/teacher_lora/$SLURM_JOB_ID" || true
fi
echo "[teacher_lora] done"
