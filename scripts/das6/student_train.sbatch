#!/usr/bin/env bash
#SBATCH --job-name=flexslm_student_train
#SBATCH --partition=defq
#SBATCH --nodelist=node205
#SBATCH --gres=gpu:A40:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=jobs/%x-%j.out
# Trains one config of the intern's code ($SH/code, branch mednest-ablations-das6) on node205
# (2 x A40, DDP, one task per GPU, all 64 CPUs). Her code keeps checkpoints and the trained model
# under $HOME/FlexSLM/{checkpoints,saved_models}, so HOME is pointed at $SH/home for the python
# processes only (export_student.sbatch reads the model from there). No W&B (her account).
# Pass --time as 2x a realistic estimate (run 4: 50 min on 4 x H100; here roughly 10x slower).
#   sbatch [--time=HH:MM:SS] scripts/das6/student_train.sbatch sft.medmix_qwen3_1p7b_10L_kv_nokd
# Smoke test (N steps, longest examples first, throw-away HOME, no resume):
#   sbatch --time=01:00:00 --export=ALL,STUDENT_SMOKE_STEPS=12 scripts/das6/student_train.sbatch <config>
CONFIG=${1:?"usage: sbatch scripts/das6/student_train.sbatch <student config>"}
SH=/var/scratch/$USER/flexslm_student
source scripts/das6/common.sh gpu
export WANDB_MODE=disabled
RUN_HOME=$SH/home
EXTRA=""
if [[ -n "${STUDENT_SMOKE_STEPS:-}" ]]; then
  RUN_HOME=$SH/smoke_home/$SLURM_JOB_ID
  mkdir -p "$RUN_HOME"
  EXTRA="--no-resume"
  trap 'rm -rf "$RUN_HOME"' EXIT
fi
cd "$SH/code"
git log -1 --format="[student_train] code %H (%D)"
echo "[student_train] config=flexllama,$CONFIG home=$RUN_HOME smoke=${STUDENT_SMOKE_STEPS:-no}"
HOME=$RUN_HOME srun python run_experiment.py run "flexllama,$CONFIG" $EXTRA
ls -la "$RUN_HOME/FlexSLM/saved_models/" || true
echo "[student_train] done"
