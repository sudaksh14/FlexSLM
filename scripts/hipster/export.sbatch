#!/usr/bin/env bash
#SBATCH --job-name=flexslm_export
#SBATCH --partition=capacity
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 30 min on CPU for a 1.7B model, three levels.
#SBATCH --time=01:00:00
#SBATCH --output=jobs/%x-%j.out
# Exports every level of a trained config to hf_export/<readable name>/level<i> (CPU job).
#   sbatch scripts/hipster/export.sbatch <config>
CONFIG=${1:?"usage: sbatch scripts/hipster/export.sbatch <config>"}
source scripts/hipster/common.sh cpu
python -m training.export_flex_hf "flexllama,$CONFIG"
