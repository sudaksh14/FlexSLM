#!/usr/bin/env bash
#SBATCH --job-name=flexslm_continuum
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node205,node208
# node208 stays free for another user (user, 2026-11-17); use it only when the user says so.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 40 min: ~50 sub-networks x 640 x 1,024 PMC val tokens, 0.6B.
#SBATCH --time=01:30:00
#SBATCH --output=jobs/%x-%j.out
# Continuum-of-widths PMC val perplexity (eval/eval_continuum.py).
#   sbatch scripts/das6/eval_continuum.sbatch <Stage-1 config> [extra args, e.g. --init]
#   sbatch scripts/das6/eval_continuum.sbatch --hf <hf model dir or id> [extra args]
source scripts/das6/common.sh gpu
if [[ "$1" == "--hf" ]]; then
  OUT="results/pmcppl.$(echo "$2" | tr '/' '-').json"
  python -m eval.eval_continuum --hf "$2" --out "$OUT" "${@:3}"   # e.g. --batch-size 2 for a 7-8B model on an A10
else
  CFG=${1:?"usage: sbatch scripts/das6/eval_continuum.sbatch <config> [args] | --hf <model>"}
  shift
  SUFFIX=""; [[ " $* " == *" --init "* ]] && SUFFIX=".init"
  python -m eval.eval_continuum --config "$CFG" --out "results/continuum.$CFG$SUFFIX.json" "$@"
fi
