#!/usr/bin/env bash
#SBATCH --job-name=flexslm_wandb
#SBATCH --partition=defq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# W&B runs through the API: expected 30-60 min; 3 h limit. No GPU; defq because fatq is full
# (pass --nodelist at submission: node208 only when the user says so (2026-11-18), node207 otherwise; never node205/206).
#SBATCH --time=03:00:00
#SBATCH --output=jobs/%x-%j.out
# Moves the per-config W&B projects into the single project MedNest: copy, verify, back up, delete (scripts/wandb_to_mednest.py).
#   sbatch --nodelist=node208 scripts/das6/wandb_to_mednest.sbatch [flags: see scripts/wandb_to_mednest.py]  (node208 only because the user said so, 2026-11-18)
source scripts/das6/common.sh cpu
python scripts/wandb_to_mednest.py "$@"
