#!/usr/bin/bash
#SBATCH --job-name=export
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

# Exports every level of a trained FlexLLaMA (saved_models/<config>.pt) as a
# Hugging Face model in hf_export/<config>/ (export_flex_hf.py).
#   sbatch export.sh flexllama,sft.medmix_qwen3_1p7b_teacher

CONFIG=${1:?"usage: sbatch export.sh <config>"}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"
python3 export_flex_hf.py "$CONFIG"
