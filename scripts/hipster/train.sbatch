#!/usr/bin/env bash
#SBATCH --job-name=flexslm_train
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=jobs/%x-%j.out
# Trains one config on hipster (performance: 4 x RTX 6000 Ada 48 GB, DDP one task per GPU, 32 CPUs per
# GPU as the partition enforces; no --exclusive on this shared cluster). Pass --time as 2x a
# pessimistic estimate (max 7 days).
#   sbatch [--time=HH:MM:SS] scripts/hipster/train.sbatch <config, e.g. s1.qwen3_1p7b.full>
# Smoke test (N steps, throw-away store, no W&B, no resume):
#   sbatch --time=02:00:00 --export=ALL,SMOKE_STEPS=20 scripts/hipster/train.sbatch <config>
CONFIG=${1:?"usage: sbatch scripts/hipster/train.sbatch <config>"}
EXTRA=""
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  export FLEXSLM_STORE=/scratch/$USER/flexslm/smoke_store/$SLURM_JOB_ID
  export FLEXSLM_SMOKE_STEPS=$SMOKE_STEPS FLEXSLM_SMOKE_LONGEST=1 WANDB_MODE=disabled
  EXTRA="--no-resume"
fi
source scripts/hipster/common.sh gpu
echo "[train] config=flexllama,$CONFIG smoke=${SMOKE_STEPS:-no} store=$FLEXSLM_STORE"
srun python -m training.run_experiment run "flexllama,$CONFIG" $EXTRA
if [[ -n "${SMOKE_STEPS:-}" ]]; then rm -rf "$FLEXSLM_STORE"; fi
echo "[train] done"
