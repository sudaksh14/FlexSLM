#!/usr/bin/env bash
#SBATCH --job-name=flexslm_student_pytest
#SBATCH --partition=fatq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected (pessimistic) 10 min: CPU unit tests (tiny models) in the intern's code.
#SBATCH --time=00:30:00
#SBATCH --output=jobs/%x-%j.out
# Unit tests of the intern's code clone ($SH/code, branch mednest-ablations-das6) as a batch job.
#   sbatch scripts/das6/student_pytest.sbatch test/test_sft_microbatch.py
SH=/var/scratch/$USER/flexslm_student
source scripts/das6/common.sh cpu
cd "$SH/code"
git log -1 --format="[student_pytest] code %H (%D)"
python -m pytest "$@" -q -p no:cacheprovider
echo "[student_pytest] PASSED"
