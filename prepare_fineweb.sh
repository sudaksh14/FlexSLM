#!/usr/bin/bash
#SBATCH --job-name=prepare_fineweb
#SBATCH --partition=capacity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --output=./jobs/%x-%j.out

# One-off tokenization of FineWeb-Edu into flat .bin files for the memmap data
# pipeline. Run once per tokenizer:
#
#   sbatch prepare_fineweb.sh JackFram/llama-160m
#   sbatch prepare_fineweb.sh Qwen/Qwen2.5-1.5B 1.4e9
#
# $1 = HF tokenizer name (default llama-160m), $2 = training tokens (default 1.4e9).
# CPU-only work, but the partition requires a GPU allocation, so it takes 1.
# Output goes to <DATA_PATH>/fineweb_bin/<tokenizer-slug>/ on shared storage so
# every later job can read it regardless of which node it lands on.

TOKENIZER=${1:-JackFram/llama-160m}
MAX_TOKENS=${2:-1.4e9}
SAMPLE=${3:-sample-10BT}   # sample-100BT for horizons above 10B tokens

LOCAL_CACHE=/local_scratch/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL_CACHE"
trap 'rm -rf "$LOCAL_CACHE"' EXIT
export HF_DATASETS_CACHE=$LOCAL_CACHE/hf_cache
export HF_HOME=$LOCAL_CACHE/hf_home

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"
echo "Preparing FineWeb-Edu ($SAMPLE) for tokenizer=$TOKENIZER max_tokens=$MAX_TOKENS"
python3 prepare_fineweb.py --tokenizer "$TOKENIZER" --max-tokens "$MAX_TOKENS" --sample "$SAMPLE"
