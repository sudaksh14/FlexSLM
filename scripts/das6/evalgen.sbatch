#!/usr/bin/env bash
#SBATCH --job-name=flexslm_evalgen
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node205,node208
# node208 stays free for another user (user, 2026-11-17); use it only when the user says so.
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
PF=${PROMPT_FORMAT:-as_is}
# Optional (environment): BUDGETS=64,128,... scores every smaller decoding budget from the same greedy
# generation; PER_SOURCE=N keeps the first N questions of each test set (deterministic subset);
# SAMPLE=qwen3 samples with Qwen3's thinking-mode settings (T 0.6, top-p 0.95, top-k 20, seed 0).
source scripts/das6/common.sh gpu
PFTAG=""; [[ "$PF" != as_is ]] && PFTAG=".p$PF"
NTAG=""; [[ -n "${PER_SOURCE:-}" ]] && NTAG=".n$PER_SOURCE"
SAMPLING=""; [[ "${SAMPLE:-}" == qwen3 ]] && { SAMPLING="--temperature 0.6 --top_p 0.95 --top_k 20 --seed 0"; NTAG="$NTAG.sq3"; }
OUT="results/gen.$(echo "$MODEL" | tr '/' '-').t$MAX_NEW$PFTAG$NTAG${SRC:+.$(echo "$SRC" | tr '/,' '-+')}.json"
python -m eval.eval_gen "$MODEL" --out "$OUT" --max_new_tokens "$MAX_NEW" --batch_size "$BS"   --prompt_format "$PF" ${SRC:+--sources "$SRC"} ${REVISION:+--revision "$REVISION"}   ${TRUST_REMOTE_CODE:+--trust_remote_code} ${BUDGETS:+--budgets "$BUDGETS"}   ${PER_SOURCE:+--per_source "$PER_SOURCE"} $SAMPLING
