#!/usr/bin/env bash
#SBATCH --job-name=flexslm_pytest
#SBATCH --partition=defq
#SBATCH --nodelist=node207
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 10 min: CPU unit tests (tiny models). On node207 because fatq is fully occupied by
# another user's long jobs; node208 stays free (user, 2026-11-17).
#SBATCH --time=00:30:00
#SBATCH --output=jobs/%x-%j.out
# Unit tests as a batch job (never on the login node).
#   sbatch scripts/das6/pytest.sbatch test/test_kd_sweep.py test/test_continuum.py ...
source scripts/das6/common.sh gpu
python -m pytest "$@" -q -p no:cacheprovider
echo "[pytest] PASSED"
