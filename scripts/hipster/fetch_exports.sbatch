#!/usr/bin/env bash
#SBATCH --job-name=flexslm_fetch
#SBATCH --partition=capacity
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
# Expected (pessimistic) 40 min: ~12 GB streamed from DAS-6, one connection at a time (DAS-6 throttles bursts).
#SBATCH --time=01:30:00
#SBATCH --output=jobs/%x-%j.out
# Copies HF exports the hipster runs need from DAS-6 into hf_export/ (on /scratch: reproducible items only).
#   sbatch scripts/hipster/fetch_exports.sbatch <export dir relative to hf_export/> ...
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/FlexSLM-medbench}"
D6=fs2.das6.science.uva.nl
for d in "$@"; do
  if [[ -f "hf_export/$d/.complete" ]]; then echo "[fetch] $d already there"; continue; fi
  echo "[fetch] $d"; mkdir -p "hf_export/$d"
  ssh -o BatchMode=yes "$D6" "tar -cf - -C /var/scratch/$USER/flexslm/hf_export/$d ." | tar -xf - -C "hf_export/$d"
  touch "hf_export/$d/.complete"
done
du -sh hf_export/*
echo "[fetch] done"
