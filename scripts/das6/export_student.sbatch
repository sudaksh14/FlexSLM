#!/usr/bin/env bash
#SBATCH --job-name=flexslm_export_student
#SBATCH --partition=fatq
#SBATCH --nodelist=node202
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# Expected (pessimistic) 40 min on CPU: 10 levels of a 1.7B model.
#SBATCH --time=01:30:00
#SBATCH --output=jobs/%x-%j.out
# Exports every level of one of the intern's 10-level Qwen3-1.7B checkpoints with HER code
# (github berraylvc/flexSlm_uva @1e5fdfa, cloned to $SH/code): her levels use kv_group slicing
# (e.g. 5 query / 3 kv heads), which her exporter maps to standard HF Qwen3 by repeating the kv
# head of every query head. Evaluation then uses OUR eval/eval_gen.py (evalgen.sbatch).
# Her code reads $HOME/FlexSLM/saved_models/<name>.pt on a cluster, so HOME is pointed at
# $SH/home for the python process only (our ~/FlexSLM/saved_models is never touched).
#   sbatch scripts/das6/export_student.sbatch sft.medmix_qwen3_1p7b_10L_kv_teacher
CONFIG=${1:?"usage: sbatch scripts/das6/export_student.sbatch <student config>"}
SH=/var/scratch/$USER/flexslm_student
source scripts/das6/common.sh cpu
OUT=$FLEXSLM_SCRATCH/hf_export/student__$CONFIG
cd "$SH/code"
git log -1 --format="[export_student] code %H"
HOME=$SH/home python export_flex_hf.py "flexllama,$CONFIG" --out "$OUT"
echo "[export_student] done -> $OUT"
