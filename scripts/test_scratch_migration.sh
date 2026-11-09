#!/usr/bin/bash
#SBATCH --job-name=test_scratch
#SBATCH --partition=capacity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --mem=60G
#SBATCH --output=./jobs/%x-%j.out

# Validates the memmap data pipeline (utils.load_fineweb_edu_memmap) against
# synthetic .bin files - no FineWeb download needed.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"
# repo root on the path: running test/x.py puts test/ on sys.path, not the root
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
python3 test/test_scratch_migration.py
