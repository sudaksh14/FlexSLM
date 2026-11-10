#!/usr/bin/env bash
#SBATCH --job-name=flexslm_prepare_pmc
#SBATCH --partition=fatq
#SBATCH --nodelist=node202
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# Expected (pessimistic) 2 h for 1B tokens (~65k articles at ~0.6 s each over 64
# parallel downloads, plus tokenization); limit 2x that.
#SBATCH --time=04:00:00
#SBATCH --output=jobs/%x-%j.out
# Builds data/pmc_qwen3/{val,test,train}.bin + meta.json for Stage 1 (CPU job).
#   sbatch scripts/das6/prepare_pmc.sbatch [train_tokens, default 1e9]
TRAIN_TOKENS=${1:-1e9}
source scripts/das6/common.sh cpu
python -m training.prepare_pmc --train-tokens "$TRAIN_TOKENS" --out-dir data/pmc_qwen3
