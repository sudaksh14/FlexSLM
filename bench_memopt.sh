#!/usr/bin/bash
#SBATCH --job-name=bench_memopt
#SBATCH --partition=performance
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --mem=100G
#SBATCH --output=./jobs/%x-%j.out

# A/B peak-memory comparison of the original vs memory-optimized KD trainer on
# the real Qwen2.5-1.5B config. Single GPU is enough: DDP replicates per rank.

LOCAL_CACHE=/local_scratch/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL_CACHE"
trap 'rm -rf "$LOCAL_CACHE"' EXIT
export HF_DATASETS_CACHE=$LOCAL_CACHE/hf_cache
export HF_HOME=$LOCAL_CACHE/hf_home

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"
python3 bench_memopt.py
