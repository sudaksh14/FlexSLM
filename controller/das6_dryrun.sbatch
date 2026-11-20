#!/usr/bin/env bash
#SBATCH --job-name=flexslm_ctrl_dry
#SBATCH --partition=defq
#SBATCH --nodelist=node207
#SBATCH --gres=gpu:A10:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
# ~15 min expected; 40 min limit. node207 (1 x A10, eval node; node205/206 train, node208 is not ours to use).
#SBATCH --time=00:40:00
#SBATCH --output=jobs/%x-%j.out
# Dry run of controller/jetson_runner.py on a DAS-6 GPU before the board (no rails, no nvpmodel: energy falls
# back to the profile model, phases use `keep`). Trained M3 export, smoke bundle, GPU-load phase in the middle.
#   1) --score cluster  (the board's default path)       2) --score device (board answers vs cluster answers)
# Latencies are an A10's, not the Jetson's: the 15 W profile over-predicts them, which also exercises the
# feedback factor rho. Only the mechanics are validated here, not any number.
#   bash scripts/das6/sync.sh && ssh das6 'cd ~/FlexSLM && sbatch controller/das6_dryrun.sbatch'
EXP=${1:-hf_export/flexllama__sft.paper.qwen3_0p6b_nested}
source scripts/das6/common.sh gpu
export MEDNEST_OUT=/var/scratch/$USER/flexslm/controller_dryrun/$SLURM_JOB_ID
mkdir -p "$MEDNEST_OUT"
COMMON=(--cooldown-c 0 --power-source none --idle-s 1 --bundle controller/bundles/smoke.json --elastic "$EXP"
        --profile controller/bundles/profile.0p6b.15W.jsonl --phases keep:14,keep+gpu:14,keep:14)
set -x
python controller/test_controller.py
python controller/jetson_runner.py "${COMMON[@]}" --tag cluster --score cluster \
  --policies fixed:max fixed:min deadline deadline+fb queue:0.5 queue-task:0.5
python controller/jetson_runner.py "${COMMON[@]}" --tag device --score device --policies fixed:max deadline+fb
python controller/jetson_runner.py "${COMMON[@]}" --tag nodl --score cluster --no-deadlines \
  --policies fixed:max queue-task:0.5
for f in "$MEDNEST_OUT"/controller_*.jsonl; do python controller/analyze_board.py "$f" --out "$f.summary.json"; done
ls -la "$MEDNEST_OUT"
