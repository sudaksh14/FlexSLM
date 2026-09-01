#!/usr/bin/bash
#SBATCH --job-name=flexllama_choose_slm
#SBATCH --partition=performance
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=150:00:00
#SBATCH --mem=300G
#SBATCH --output=./jobs/%x-%j.out

# Pick which SLM to train by uncommenting exactly ONE of the CONFIG= lines
# below, then: sbatch submit_flexllama_choose_slm.sh
#
# Uncomment exactly one - if more than one is uncommented, bash just keeps
# reassigning CONFIG top to bottom, so the last uncommented line silently
# wins. The check below only catches the "none uncommented" case.

# --- llama-160m (random-init unless noted; small enough to iterate fast) ---
# CONFIG=flexllama,fineweb.kd_lambda05                        # mixed KD (kd_lambda=0.5), warm-started
# CONFIG=flexllama,fineweb.kd_lambda05_1p4B                   # same, larger ~1.4B-token FineWeb-Edu slice
# CONFIG=flexllama,fineweb.kd_lambda05_5levels               # same, 5 flex levels

# --- TinyLlama-1.1B (GQA 32/4 heads) ---
# CONFIG=flexllama,fineweb.kd_tinyllama_warmstart              # warm-started, 3 levels (default)
# CONFIG=flexllama,fineweb.kd_tinyllama_warmstart_5levels    # warm-started, 5 levels
# CONFIG=flexllama,fineweb.kd_tinyllama_warmstart_7levels    # warm-started, 7 levels
# CONFIG=flexllama,fineweb.kd_tinyllama_warmstart_10levels   # warm-started, 10 levels
# CONFIG=flexllama,fineweb.kd_tinyllama_warmstart_bigteacher  # warm-started self, KD teacher=Llama-2-7b (same vocab, gated on HF)

# --- SmolLM2 (tied embeddings) ---
# CONFIG=flexllama,fineweb.kd_smollm2_135m_warmstart         # 135M, warm-started
# CONFIG=flexllama,fineweb.kd_smollm2_360m_warmstart         # 360M, warm-started
# CONFIG=flexllama,fineweb.kd_smollm2_1p7b_warmstart         # 1.7B, warm-started

# --- Qwen2.5 (tied embeddings + biased q/k/v projections) ---
# CONFIG=flexllama,fineweb.kd_qwen25_0p5b_warmstart          # 0.5B, warm-started
CONFIG=flexllama,fineweb.kd_qwen25_1p5b_warmstart           # 1.5B, warm-started

# --- Qwen2.5-Coder (same arch as Qwen2.5, code-tuned checkpoints) ---
# CONFIG=flexllama,fineweb.kd_qwen25coder_0p5b_warmstart              # 0.5B, warm-started self
# CONFIG=flexllama,fineweb.kd_qwen25coder_1p5b_warmstart              # 1.5B, warm-started self
# CONFIG=flexllama,fineweb.kd_qwen25coder_0p5b_warmstart_bigteacher   # warm-started self (0.5B), KD teacher=Qwen2.5-Coder-1.5B

# NOTE: Llama-3.2-1B/3B and Llama-3.1-8B are NOT offered here yet - all Llama-3.x
# checkpoints use the "llama3" rope_scaling extension on top of rope_theta, which
# flex_modules.rope_attention.RoPEAttention does not implement. Warm-starting one
# would copy correct weights but apply a mismatched rotary schedule. Their
# TEACHER_PRESETS entries exist for when that's implemented; no warm-start-capable
# experiment references them until then.

if [ -z "$CONFIG" ]; then
    echo "ERROR: no CONFIG selected - uncomment exactly one CONFIG= line in $0" >&2
    exit 1
fi
echo "Training config: $CONFIG"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

nvidia-smi

exec "$SLURM_SUBMIT_DIR/experiment_job.sh" "$CONFIG"
