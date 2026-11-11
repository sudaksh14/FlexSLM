#!/usr/bin/env bash
#SBATCH --job-name=flexslm_cmp_teachers
#SBATCH --partition=fatq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 25 min: load three 7 GB state dicts + the HF base, tensor-wise statistics,
# one bf16 export. Limit 2x that.
#SBATCH --time=01:00:00
#SBATCH --output=jobs/%x-%j.out
# Weight-change comparison of the student's old teacher and ours; exports the old teacher.
source scripts/das6/common.sh cpu
python scripts/compare_teachers.py "$@"
