#!/usr/bin/bash
#SBATCH --job-name=flexllama_fineweb_4gpu
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=650G
#SBATCH --output=./jobs/%x-%j.out

# 4 GPUs on the gpu_h100 partition, DDP across all 4 (nccl backend, see
# training.py's DDPStrategy + devices="auto").
#
# Usage: sbatch submit_flexllama_4gpu.sh [config_name]
# Defaults to the llama-160m KD run over the ~1.4B-token FineWeb-Edu slice.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_TOKEN=$(cat "$HOME/.hf_token")
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

nvidia-smi

CONFIG=${1:-flexllama,fineweb.kd_lambda05_1p4B}

exec "$SLURM_SUBMIT_DIR/experiment_job.sh" "$CONFIG"
