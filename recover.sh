#!/usr/bin/bash
#SBATCH --job-name=recover
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

# Writes saved_models/<config>.pt from checkpoints/<config>/best-model.ckpt and
# exports every level to hf_export/<config>/ (recover_ckpt.py). CPU work; the
# GPU partition is only used because it is the one this project runs on.
#   sbatch recover.sh flexllama,sft.medmix_qwen3_1p7b

CONFIG=${1:?"usage: sbatch recover.sh <config>"}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"
python3 recover_ckpt.py "$CONFIG" --export
