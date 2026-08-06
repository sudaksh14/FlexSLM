#!/usr/bin/bash
#SBATCH --job-name=flexllama_memopt
#SBATCH --partition=performance
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=150:00:00
#SBATCH --mem=300G
#SBATCH --output=./jobs/%x-%j.out

# Memory-optimized FlexLLaMA KD runs (training_memopt.py: chunked loss +
# activation checkpointing). Same nested-width algorithm and identical gradients
# as the training.py trainers - only peak memory differs - so results are
# directly comparable to the non-memopt configs.
#
# Pick a config by uncommenting exactly ONE CONFIG= line, then:
#   sbatch submit_flexllama_memopt.sh
# Or override from the command line:
#   sbatch submit_flexllama_memopt.sh flexllama,fineweb.kd_smollm2_1p7b_warmstart_memopt
#
# 4x RTX 6000 Ada (48GB). cpus-per-task=32 x ntasks=4 = 128 satisfies the
# performance partition's 32 CPUs/GPU ratio; mem must be explicit (not 0).

# --- 1.5B / 1.7B class: these are the ones that OOM'd without the optimization ---
CONFIG=flexllama,fineweb.kd_qwen25_1p5b_warmstart_memopt      # Qwen2.5-1.5B, batch_size=4
# CONFIG=flexllama,fineweb.kd_smollm2_1p7b_warmstart_memopt   # SmolLM2-1.7B, batch_size=4

# --- smaller models: memopt not required, but works and leaves room for larger batches ---
# CONFIG=flexllama,fineweb.kd_qwen25_0p5b_warmstart_memopt    # Qwen2.5-0.5B
# CONFIG=flexllama,fineweb.kd_tinyllama_warmstart_memopt      # TinyLlama-1.1B

CONFIG=${1:-$CONFIG}
if [ -z "$CONFIG" ]; then
    echo "ERROR: no CONFIG selected - uncomment one CONFIG= line in $0 or pass one as \$1" >&2
    exit 1
fi
echo "Training config: $CONFIG"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

# Reduces allocator fragmentation - the failing runs showed multi-GB
# "reserved but unallocated" at the point of the OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

exec "$SLURM_SUBMIT_DIR/experiment_job.sh" "$CONFIG" "${2:-}"
