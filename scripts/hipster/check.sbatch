#!/usr/bin/env bash
#SBATCH --job-name=flexslm_check
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=jobs/%x-%j.out
# Environment check on hipster (transformers 4.57 / torch 2.8 / python 3.9 vs DAS-6's 4.51 / 2.7 / 3.10):
# FlexLLaMA warm starts must reproduce the HF Qwen3-0.6B and 1.7B logits, and the PMC val ppl of
# Qwen3-1.7B must match DAS-6 (7.918, job 28156).
#   sbatch scripts/hipster/check.sbatch
source scripts/hipster/common.sh gpu
python -m scripts.check_qwen3_warmstart flexllama,sft.paper.qwen3_0p6b_conventional --no-export
python -m scripts.check_qwen3_warmstart flexllama,sft.paper.qwen3_1p7b_teacher --no-export
python -m eval.eval_continuum --hf Qwen/Qwen3-1.7B --out results/pmcppl.hipster.Qwen-Qwen3-1.7B.json
echo "[check] done"
