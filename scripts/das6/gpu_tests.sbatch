#!/usr/bin/env bash
#SBATCH --job-name=flexslm_gpu_tests
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node205
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 20 min: unit tests + two real-checkpoint warm-start checks.
#SBATCH --time=00:40:00
#SBATCH --output=jobs/%x-%j.out
# GPU checks before any training: unit tests, then the Qwen3-0.6B and Qwen3-1.7B
# (GQA-preserving levels) warm starts must reproduce Hugging Face's logits.
#   sbatch scripts/das6/gpu_tests.sbatch      (from ~/FlexSLM)
source scripts/das6/common.sh gpu
python -m pytest test/test_level_weights.py test/test_importance_order.py test/test_modules.py test/test_head_dim_decoupled.py test/test_memopt_equivalence.py -q -p no:cacheprovider
python -m scripts.check_qwen3_warmstart flexllama,sft.paper.qwen3_0p6b_conventional --no-export
python -m scripts.check_qwen3_warmstart flexllama,sft.paper.qwen3_1p7b_teacher --no-export
echo "[gpu_tests] ALL PASSED"
