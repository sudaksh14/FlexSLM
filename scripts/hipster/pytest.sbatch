#!/usr/bin/env bash
#SBATCH --job-name=flexslm_pytest
#SBATCH --partition=capacity
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# Expected (pessimistic) 10 min: CPU unit tests (tiny models), never on the login node.
#SBATCH --time=00:30:00
#SBATCH --output=jobs/%x-%j.out
#   sbatch scripts/hipster/pytest.sbatch test/test_kd_sweep.py ...
source scripts/hipster/common.sh cpu
# my_env has no pytest and must not be modified: install it into a separate directory (user, 2026-11-18).
PT=$HOME/flexslm_saves/pytest_overlay
[[ -d $PT/pytest ]] || pip install --quiet --target "$PT" pytest
PYTHONPATH="$PT${PYTHONPATH:+:$PYTHONPATH}" python -m pytest "$@" -q -p no:cacheprovider
echo "[pytest] PASSED"
