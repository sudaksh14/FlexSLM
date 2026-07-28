#!/usr/bin/bash
#SBATCH --job-name=flexllama_fineweb_4gpu
#SBATCH --partition=capacity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=180G
#SBATCH --output=%x-%j.out

# 4x RTX 6000 Ada on the `performance` partition, DDP across all 4 (nccl backend,
# see training.py's DDPStrategy + devices="auto"). cpus-per-task=32 x ntasks=4 = 128
# satisfies the partition's required 32 CPUs/GPU ratio.

# 4x L4 on the `capacity` partition, DDP across all 4 (nccl backend,
# see training.py's DDPStrategy + devices="auto"). cpus-per-task=16 x ntasks=4 = 64
# satisfies the partition's required 16 CPUs/GPU ratio. mem is explicit (not 0) -
# this cluster's scheduler rejects --mem=0 on multi-GPU/multi-task submissions.
#
# Usage: sbatch submit_flexllama_4gpu.sh [config_name]
# Defaults to the llama-160m KD run over the ~1.4B-token FineWeb-Edu slice.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

CONFIG=${1:-flexllama,fineweb.kd_lambda05_1p4B}

exec "$(dirname "$0")/experiment_job.sh" "$CONFIG"
