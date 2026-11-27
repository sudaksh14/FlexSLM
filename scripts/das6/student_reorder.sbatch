#!/usr/bin/env bash
#SBATCH --job-name=flexslm_student_reorder
#SBATCH --partition=defq
#SBATCH --nodelist=node205
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
# Expected (pessimistic) 20 min. On node205 because node206/207 are held by another user's jobs and
# the ablation trainings that follow run on node205 anyway (node208 stays free, user 2026-11-17).
#SBATCH --time=01:00:00
#SBATCH --output=jobs/%x-%j.out
# Rebuilds the intern's importance-reordered warm start of Qwen3-1.7B (hidden + MLP channels,
# reorder_qwen3.py, calibration = 512 medmix_sft training examples, seed 0) in $SH/code/hf_export/
# qwen3_1p7b_reordered, the start of _10L_kv_teacher (run 4). The copy on DAS-6 had only config +
# tokenizer. Fails unless the result matches her logs (Snellius reorder-29104870/29106477).
#   sbatch scripts/das6/student_reorder.sbatch
SH=/var/scratch/$USER/flexslm_student
source scripts/das6/common.sh gpu
cd "$SH/code"
git log -1 --format="[student_reorder] code %H (%D)"
LOG=$(mktemp)
python reorder_qwen3.py --force 2>&1 | tee "$LOG"
# Her script @1e5fdfa no longer prints the channel-overlap line of her logs; check the reordered
# untrained losses instead (her log, level 4 / 6 / 7 of the 10-level model: 7.935 / 6.330 / 3.961).
for v in 7.935 6.330 3.961; do
  grep -q " $v" "$LOG" \
    || { echo "[student_reorder] ERROR: ordering differs from the intern's run-4 warm start ($v missing)" >&2; exit 3; }
done
ls -la hf_export/qwen3_1p7b_reordered
echo "[student_reorder] done, matches the intern's ordering"
