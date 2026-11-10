#!/usr/bin/env bash
#SBATCH --job-name=flexslm_diag_order
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node205
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 20 min: 8 variants x (save 2.4 GB + 3 level ppl passes on 16k tokens).
#SBATCH --time=00:45:00
#SBATCH --output=jobs/%x-%j.out
# M1 diagnostic (scripts/diag_ordering.py): which part of the ordering hurts the sub-levels.
source scripts/das6/common.sh gpu
export TMPDIR=/var/scratch/$USER/flexslm/tmp_$SLURM_JOB_ID; mkdir -p $TMPDIR
python scripts/diag_ordering.py
rm -rf $TMPDIR
