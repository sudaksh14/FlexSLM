#!/usr/bin/bash
#SBATCH --job-name=test_inplace_kd
#SBATCH --partition=capacity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --mem=60G
#SBATCH --output=./jobs/%x-%j.out

# CPU-only unit tests for in-place level-to-level KD (inplace_kd) plus import smoke tests for the
# relocated training/scripts/eval/data packages. Submit from the repo root.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"
python3 -m unittest test.test_inplace_kd test.test_imports -v
