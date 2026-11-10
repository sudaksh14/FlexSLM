#!/usr/bin/env bash
#SBATCH --job-name=flexslm_order
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node205
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 30 min: 512 x 512 calibration tokens (jobs 28091/28094 ran out of
# memory: a hook reference cycle kept every graph alive, fixed) forward+backward through
# Qwen3-0.6B in fp32, then two per-level PMC val ppl passes. Limit 2x that.
#SBATCH --time=01:00:00
#SBATCH --output=jobs/%x-%j.out
# M1: Taylor importance ordering of Qwen3-0.6B -> hf_export/qwen3-0.6b.taylor
# (training/importance_order.py). Needs data/pmc_qwen3 (prepare_pmc.sbatch).
#   sbatch scripts/das6/importance_order.sbatch [extra args for training.importance_order]
source scripts/das6/common.sh gpu
python -m training.importance_order --model Qwen/Qwen3-0.6B --out hf_export/qwen3-0.6b.taylor --batch-size 1 --seq-len 512 --calib-seqs 512 "$@"
