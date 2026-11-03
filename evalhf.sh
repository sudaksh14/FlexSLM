#!/usr/bin/bash
#SBATCH --job-name=evalhf
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

# Baseline: evaluates an unmodified Hugging Face checkpoint (no FlexSLM, no
# training) on the same cloze-format medical tasks used for our own models.
#
# Usage:
#   sbatch evalhf.sh <hf_model> [tasks] [batch_size]
#
#   sbatch evalhf.sh Qwen/Qwen2.5-0.5B
#   sbatch evalhf.sh Qwen/Qwen3-1.7B medcloze 8

MODEL=${1:?"usage: sbatch evalhf.sh <hf_model> [tasks] [batch_size]"}
TASKS=${2:-medcloze}
BATCH_SIZE=${3:-16}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"

OUT="results/hf.$(echo "$MODEL" | tr '/' '-').$TASKS"
echo "[evalhf] model=$MODEL tasks=$TASKS batch_size=$BATCH_SIZE output=$OUT"
lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks "$TASKS" \
    --include_path medcloze \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUT" \
    --log_samples
