#!/usr/bin/bash
#SBATCH --job-name=fetch_pmc_ids
#SBATCH --partition=staging
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=./jobs/%x-%j.out

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

python3 fetch_pmc_commercial_ids.py
