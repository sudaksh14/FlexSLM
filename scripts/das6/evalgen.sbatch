#!/usr/bin/env bash
#SBATCH --job-name=flexslm_evalgen
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node205
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 45 min: 12,797 test prompts, 32 new tokens, <=1.7B model.
#SBATCH --time=01:30:00
#SBATCH --output=jobs/%x-%j.out
# Answer-only generated evaluation on the seven FaithMed test sets (eval/eval_gen.py).
#   sbatch scripts/das6/evalgen.sbatch <hf model dir or id> [max_new_tokens] [batch_size] [sources]
# sources: comma-separated subset of the test sets (e.g. openlifescienceai/medqa); long
# generation budgets (untrained models reason before answering) also need a longer --time.
MODEL=${1:?"usage: sbatch scripts/das6/evalgen.sbatch <hf_model> [max_new_tokens] [batch_size]"}
MAX_NEW=${2:-32}
BS=${3:-32}
SRC=${4:-}
source scripts/das6/common.sh gpu
OUT="results/gen.$(echo "$MODEL" | tr '/' '-').t$MAX_NEW${SRC:+.$(echo "$SRC" | tr '/,' '-+')}.json"
python -m eval.eval_gen "$MODEL" --out "$OUT" --max_new_tokens "$MAX_NEW" --batch_size "$BS" ${SRC:+--sources "$SRC"}
