#!/usr/bin/bash
#SBATCH --job-name=evalgen
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

# FaithMed-format generated-answer evaluation (eval_gen.py) of one HF model
# directory, e.g. an exported FlexLLaMA level:
#   sbatch evalgen.sh hf_export/<config>/level2 [max_new_tokens] [batch_size]

MODEL=${1:?"usage: sbatch evalgen.sh <hf_model_dir> [max_new_tokens] [batch_size]"}
MAX_NEW=${2:-32}
BATCH_SIZE=${3:-32}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"

OUT="results/gen.$(echo "$MODEL" | tr '/' '-').t$MAX_NEW.json"
echo "[evalgen] model=$MODEL max_new_tokens=$MAX_NEW batch_size=$BATCH_SIZE output=$OUT"
python3 eval_gen.py "$MODEL" --out "$OUT" --max_new_tokens "$MAX_NEW" --batch_size "$BATCH_SIZE"
