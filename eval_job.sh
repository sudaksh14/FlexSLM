#!/usr/bin/bash
#SBATCH --job-name=eval_job
#SBATCH --partition=performance
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

# Evaluates a completed FlexGPT/FlexLLaMA checkpoint (saved_models/<config>.pt,
# written by utils.save_model() when a training run finishes). Two modes:
#
#   wiki - fast per-level WikiText-103 perplexity (eval_wikitext103.py). A quick
#          sanity signal to cross-check W&B validation numbers - out-of-
#          distribution for FineWeb-Edu-trained models, not a benchmark result.
#   lm   - full lm-evaluation-harness suite (eval_lm_harness.py): wikitext,
#          lambada_openai, hellaswag, piqa, arc_easy/challenge, openbookqa,
#          triviaqa by default. Needs the `lm_eval` package (pip installed into
#          my_env; note: it pins datasets==3.6.0, downgraded from 4.5.0 - the
#          streaming/memmap FineWeb-Edu pipeline was re-verified against that
#          exact version and is unaffected).
#          One flex level of the full 8-task default suite took ~3h on a 3-level
#          model (measured on a 160M-max-width FlexLLaMA) - budget --time
#          accordingly for models with more levels, or narrow --tasks.
#
# Usage:
#   sbatch eval_job.sh wiki <config> [dataset] [batch_size]
#   sbatch eval_job.sh lm   <config> [tasks] [batch_size]
#
#   sbatch eval_job.sh wiki flexllama,fineweb.kd_lambda05_1p4B
#   sbatch eval_job.sh lm   flexllama,fineweb.kd_lambda05_1p4B lambada_openai,hellaswag 4

MODE=${1:?"usage: sbatch eval_job.sh <wiki|lm> <config> [...]"}
CONFIG=${2:?"usage: sbatch eval_job.sh <wiki|lm> <config> [...]"}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"

case "$MODE" in
  wiki)
    DATASET=${3:-wikitext-103-raw-v1}
    BATCH_SIZE=${4:-8}
    echo "[eval_job:wiki] config=$CONFIG dataset=$DATASET batch_size=$BATCH_SIZE"
    python3 eval_wikitext103.py --config "$CONFIG" --dataset "$DATASET" --batch_size "$BATCH_SIZE"
    ;;
  lm)
    TASKS=${3:-"wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,openbookqa,triviaqa"}
    BATCH_SIZE=${4:-8}
    SAFE_NAME=$(echo "$CONFIG" | tr ',/' '__')
    OUT="results/${SAFE_NAME}.json"
    echo "[eval_job:lm] config=$CONFIG tasks=$TASKS batch_size=$BATCH_SIZE output=$OUT"
    python3 eval_lm_harness.py --config "$CONFIG" --tasks "$TASKS" --batch_size "$BATCH_SIZE" --output "$OUT"
    ;;
  *)
    echo "ERROR: unknown mode '$MODE' - expected 'wiki' or 'lm'" >&2
    exit 1
    ;;
esac
