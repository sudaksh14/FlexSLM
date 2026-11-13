#!/usr/bin/env bash
#SBATCH --job-name=flexslm_migrate
#SBATCH --partition=capacity
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/skalra/flexslm_migrate-%j.out
# Moves everything that must survive the periodic /scratch cleanup out of /scratch (user, 2026-11-13):
#   data/pmc_qwen3, data/medmix_sft  -> ~/FlexSLM-medbench/data/   (real directory instead of a symlink)
#   results/                         -> ~/FlexSLM-medbench/results/
#   hf_cache/ (Hugging Face cache)   -> ~/flexslm_saves/cache/huggingface/   (scripts/hipster/common.sh
#                                       switches HF_HOME to it as soon as that directory exists)
# Stays in /scratch (checkpoints and reproducible items only): store/ (checkpoints, wandb), hf_export/,
# jobs/ (Slurm logs; their numbers are recorded in implementation_log.md), smoke_store/.
# Each piece is copied, compared byte for byte (diff -rq), swapped in, and only then deleted from
# /scratch. Run it ONLY when none of our jobs are queued or running (checked below): it replaces the
# data/ and results/ symlinks that running jobs read.
#   sbatch scripts/hipster/migrate_to_home.sbatch
# Paths can be overridden for a dry run on another machine: S (scratch), R (repo), HS (saves dir).
set -euo pipefail
S=${S:-/scratch/$USER/flexslm}
R=${R:-$HOME/FlexSLM-medbench}
HS=${HS:-$HOME/flexslm_saves}

if command -v squeue >/dev/null 2>&1 && [[ -z "${SKIP_JOB_CHECK:-}" ]]; then
  others=$(squeue -u "$USER" -h -o "%i %j" | grep -E " flexslm_" | grep -v " flexslm_migrate" || true)
  if [[ -n "$others" ]]; then
    echo "REFUSING: other flexslm jobs are queued or running:"; echo "$others"; exit 3
  fi
fi

# move_dir <scratch dir> <target dir>: copy, verify, swap, delete the scratch copy.
move_dir() {
  local src=$1 dst=$2 tmp="$2.migrating"
  [[ -d "$src" ]] || { echo "[migrate] skip $src (not there)"; return 0; }
  echo "[migrate] $src -> $dst"; date
  rm -rf "$tmp"; mkdir -p "$tmp"
  cp -a "$src"/. "$tmp"/
  diff -rq "$src" "$tmp" && echo "[migrate]   verified identical"
  if [[ -L "$dst" ]]; then rm "$dst"; elif [[ -d "$dst" ]]; then
    # a real directory already there: merge, never overwrite
    cp -an "$tmp"/. "$dst"/ && rm -rf "$tmp"; rm -rf "$src"; echo "[migrate]   merged into existing $dst"; return 0
  fi
  mkdir -p "$(dirname "$dst")"; mv "$tmp" "$dst"
  rm -rf "$src"
  echo "[migrate]   done: $(du -sh "$dst" | cut -f1) in $dst"
}

df -h "$HOME" | tail -1
mkdir -p "$R/data"
for d in pmc_qwen3 medmix_sft; do
  [[ -L "$R/data" ]] && { rm "$R/data"; mkdir -p "$R/data"; }   # data/ itself was a symlink to scratch
  move_dir "$S/data/$d" "$R/data/$d"
done
rmdir "$S/data" 2>/dev/null || true
move_dir "$S/results" "$R/results"
move_dir "$S/hf_cache" "$HS/cache/huggingface"
df -h "$HOME" | tail -1
echo "[migrate] finished; repo data/ and results/ are real directories in $R, HF cache in $HS/cache/huggingface"
