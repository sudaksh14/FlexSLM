#!/usr/bin/env bash
#SBATCH --job-name=flexslm_train
#SBATCH --partition=defq
#SBATCH --nodelist=node205
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=jobs/%x-%j.out
# Trains one config on node205 (2x A40, DDP, one task per GPU). Pass --time on
# the command line as 2x a pessimistic runtime estimate.
#   sbatch [--time=HH:MM:SS] scripts/das6/train.sbatch <config, e.g. sft.paper.qwen3_1p7b_teacher>
# Smoke test (N steps, longest examples first, throw-away store, no W&B, no resume):
#   sbatch --time=01:00:00 --export=ALL,SMOKE_STEPS=20 scripts/das6/train.sbatch <config>
CONFIG=${1:?"usage: sbatch scripts/das6/train.sbatch <config>"}
EXTRA=""
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  export FLEXSLM_STORE=/var/scratch/$USER/flexslm/smoke_store/$SLURM_JOB_ID
  export FLEXSLM_SMOKE_STEPS=$SMOKE_STEPS FLEXSLM_SMOKE_LONGEST=1 WANDB_MODE=disabled
  EXTRA="--no-resume"
fi
source scripts/das6/common.sh gpu
echo "[train] config=flexllama,$CONFIG smoke=${SMOKE_STEPS:-no} store=$FLEXSLM_STORE"
srun python -m training.run_experiment run "flexllama,$CONFIG" $EXTRA
if [[ -n "${SMOKE_STEPS:-}" ]]; then rm -rf "$FLEXSLM_STORE"; fi
echo "[train] done"
