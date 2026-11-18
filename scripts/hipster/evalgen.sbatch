#!/usr/bin/env bash
#SBATCH --job-name=flexslm_evalgen
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# Pessimistic: <=1.7B model, 12,797 prompts, 32 tokens ~1 h on an L4 (24 GB). An 8B model needs
# --partition=performance --gres=gpu:rtx_6000_ada:1 --cpus-per-task=32 and a longer --time.
#SBATCH --time=02:00:00
#SBATCH --output=jobs/%x-%j.out
# Answer-only generated evaluation on the seven FaithMed test sets (eval/eval_gen.py), hipster.
#   sbatch scripts/hipster/evalgen.sbatch <hf model dir or id> [max_new_tokens] [batch_size] [sources]
MODEL=${1:?"usage: sbatch scripts/hipster/evalgen.sbatch <hf_model> [max_new_tokens] [batch_size] [sources]"}
MAX_NEW=${2:-32}
BS=${3:-16}
SRC=${4:-}
source scripts/hipster/common.sh gpu
OUT="results/gen.$(echo "$MODEL" | tr '/' '-').t$MAX_NEW${SRC:+.$(echo "$SRC" | tr '/,' '-+')}.json"
python -m eval.eval_gen "$MODEL" --out "$OUT" --max_new_tokens "$MAX_NEW" --batch_size "$BS" ${SRC:+--sources "$SRC"}
