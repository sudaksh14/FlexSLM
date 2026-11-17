#!/usr/bin/env bash
#SBATCH --job-name=flexslm_wandb
#SBATCH --partition=defq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# Copy of ~40 W&B runs through the API: expected 20-40 min; 2 h limit. No GPU; defq because fatq is full
# (pass --nodelist=node207/208 at submission, never node205/206).
#SBATCH --time=02:00:00
#SBATCH --output=jobs/%x-%j.out
# Copies finished runs into the single W&B project MedNest (scripts/wandb_to_mednest.py).
#   sbatch --nodelist=node208 scripts/das6/wandb_to_mednest.sbatch --skip-runs ouvny4cs,qp73a2io [--dry-run] [--only ID]
source scripts/das6/common.sh cpu
python scripts/wandb_to_mednest.py "$@"
