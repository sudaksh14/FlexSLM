#!/usr/bin/env bash
# Shape-only Jetson sweep at 15 W: latency, memory, energy and switch cost, with RANDOM weights of the
# MedNest shapes (no DAS-6, no trained checkpoints; answers are meaningless, accuracy is NOT reported).
#   . ~/mednest-venv/bin/activate && bash jetson/run_shape_only.sh               # full
#   QUICK=1 bash jetson/run_shape_only.sh                                          # smoke test
# Decoding is forced to the full max_new tokens (MEDNEST_FORCE_LEN=1), so every width/model decodes the
# same number of steps. Dense baselines with staged weights use them (shapes identical either way);
# 2B Apollo models are config-only (random weights built on the GPU). Not run here: the 7 W profile and the
# 15 W <-> 7 W budget trace (15 W only, by decision 2026-11-19), accuracy on the subset.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/mednest/models}
BACKBONE=${BACKBONE:-0p6b}     # 0p6b (default) | 1p7b: same protocol, shapes of the 1.7B nested model
if [[ "$BACKBONE" == 1p7b ]]; then
  R=$M/mednest-random-1p7b; SRC=$M/Qwen3-1.7B; EXTRA_EXPORT="--widths-json jetson/widths_qwen3_1p7b.json"
  SKIP_2B=1                  # the Apollo baselines are 0.6B-class; measured in the 0.6B run
else
  R=$M/mednest-random; SRC=$M/Qwen3-0.6B; EXTRA_EXPORT=""
fi
export MEDNEST_FORCE_LEN=1
export MEDNEST_OUT=${MEDNEST_OUT:-$HOME/mednest-results/shape_only_$(date +%Y%m%d_%H%M)}
mkdir -p "$MEDNEST_OUT"
exec > >(tee -a "$MEDNEST_OUT/run_shape_only.log") 2>&1

if [[ "${QUICK:-0}" == 1 ]]; then
  LIM="--limit 14"; LEVELS="0 4 8"; SW="--switches 12 --load-repeats 1"; LONG="--long-limit 1"
else
  LIM=""; LEVELS="0 1 2 3 4 5 6 7 8"; SW="--switches 60"; LONG="--long-limit 5"
fi

[[ -f "$R/level8/model.safetensors" ]] || python3 jetson/make_random_export.py --source "$SRC" --out "$R" $EXTRA_EXPORT

DENSE=()
if [[ "$BACKBONE" == 1p7b ]]; then
  SPECS=("conventional-shape=$R/level8" "indep-l0-shape=$R/level0" "indep-l1-shape=$R/level4")
else
  SPECS=("conventional-shape=$R/level8" "indep-l0-shape=$R/level0" "indep-l1-shape=$R/level4"             "apollo-0.5b=$M/Apollo-0.5B:apollo_answer:16" "apollo2-0.5b=$M/Apollo2-0.5B:apollo_answer:16"             "qwen3-0.6b-reasoning=$M/Qwen3-0.6B:as_is:1024")
fi
for spec in "${SPECS[@]}"; do
  dir=${spec#*=}; dir=${dir%%:*}
  [[ -f "$dir/config.json" ]] && DENSE+=(--dense "$spec") || echo "(skipping ${spec%%=*}: $dir not staged)"
done

echo "== 1/3 preflight (15W)"
python3 jetson/check_env.py --power-mode 15W --elastic "$R"
echo "== 2/3 widths + dense baselines at 15W (shape-only)"
python3 jetson/bench_widths.py --power-mode 15W --tag 15W --elastic "$R" --levels $LEVELS $LIM $LONG "${DENSE[@]}"
if [[ "${SKIP_2B:-0}" != 1 ]]; then
echo "== 2b/3 2B Apollo models, config-only, one process each (5 GB: nothing else may be resident)"
for n in Apollo-2B Apollo2-2B; do
  python3 jetson/drop_file_cache.py "$M" ~/mednest-results ~/.cache "$VIRTUAL_ENV"   # page cache would block the 5 GB allocation
  python3 jetson/bench_widths.py --power-mode 15W --tag 15W_$n $LIM --dense "${n,,}=$M/$n:apollo_answer:16"
done
fi
echo "== 3/3 switch cost (15W)"
python3 jetson/bench_switch.py --power-mode 15W --elastic "$R" --levels 0 4 8 $SW
sudo -n nvpmodel -m 0 >/dev/null 2>&1 || true   # back to the default mode
echo "done: $MEDNEST_OUT"
