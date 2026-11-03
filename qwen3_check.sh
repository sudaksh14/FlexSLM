#!/usr/bin/bash
#SBATCH --job-name=qwen3_check
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

# Real-checkpoint check of the Qwen3 warm start (test_qwen3_warmstart.py):
# FlexLLaMA max level vs Hugging Face Qwen3-1.7B logits, then the untrained
# levels exported to hf_export/<config>.init/ for the pre-SFT baselines.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"
python3 test_qwen3_warmstart.py
