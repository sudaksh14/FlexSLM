#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=flexslm_evalgen
#SBATCH --partition=defq
#SBATCH --exclude=node205,node206,node208
#SBATCH --time=01:30:00
#SBATCH --output=jobs/%x-%j.out
#   sbatch --export=ALL,BS=8 scripts/das6/apollo18_eval.sbatch 16 [sources]   (batch 32 OOMs on an A10: job 28321)
# Apollo-1.8B (Qwen-1 custom code, reviewed: = official Qwen/Qwen-1_8B code + a gradient-checkpointing
# method): the dot in the repo name breaks transformers dynamic modules, so the pinned commit is
# downloaded to a dot-free local directory first. Args: <max_new_tokens> [sources]
source scripts/das6/common.sh gpu
REV=73072ac1c8ec6200f50de8f48a8019fd787faa33
DIR=hf_export/apollo_1p8b
python -c "from huggingface_hub import snapshot_download as s; s(\"FreedomIntelligence/Apollo-1.8B\", revision=\"$REV\", local_dir=\"$DIR\")"
SRC=${2:-}
OUT="results/gen.FreedomIntelligence-Apollo-1.8B.t$1.papollo_answer${SRC:+.$(echo "$SRC" | tr "/," "-+")}.json"
python -m eval.eval_gen "$DIR" --out "$OUT" --max_new_tokens "$1" --batch_size ${BS:-32} --prompt_format apollo_answer \
  --trust_remote_code --revision "$REV" ${SRC:+--sources "$SRC"}
