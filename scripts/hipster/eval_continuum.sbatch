#!/usr/bin/env bash
#SBATCH --job-name=flexslm_continuum
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=jobs/%x-%j.out
# PMC val perplexity on hipster (eval/eval_continuum.py), e.g. of a candidate teacher:
#   sbatch scripts/hipster/eval_continuum.sbatch --hf Intelligent-Internet/II-Medical-8B
#   sbatch scripts/hipster/eval_continuum.sbatch <Stage-1 config> [extra args]
source scripts/hipster/common.sh gpu
if [[ "$1" == "--hf" ]]; then
  python -m eval.eval_continuum --hf "$2" --out "results/pmcppl.$(echo "$2" | tr '/' '-').json" --batch-size 4
else
  CFG=${1:?"usage: sbatch scripts/hipster/eval_continuum.sbatch <config> [args] | --hf <model>"}
  shift
  python -m eval.eval_continuum --config "$CFG" --out "results/continuum.$CFG.json" "$@"
fi
