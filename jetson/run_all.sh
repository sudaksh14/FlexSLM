#!/usr/bin/env bash
# The full MedNest Jetson sweep, in order. Run on the board inside tmux, from the repo root:
#   . ~/mednest-venv/bin/activate && bash jetson/run_all.sh            # ~2 h (estimate)
#   QUICK=1 bash jetson/run_all.sh                                      # ~10 min smoke test
# Results: $MEDNEST_OUT (default ~/mednest-results), one .jsonl per step + derived/ tables.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/mednest/models}
export MEDNEST_OUT=${MEDNEST_OUT:-$HOME/mednest-results/$(date +%Y%m%d_%H%M)}
mkdir -p "$MEDNEST_OUT"
exec > >(tee -a "$MEDNEST_OUT/run_all.log") 2>&1

if [[ "${QUICK:-0}" == 1 ]]; then
  LIM="--limit 14"; LEVELS="0 4 8"; SW="--switches 12 --load-repeats 1"; PH="15W:12,7W:12"; LONG="--long-limit 1"
else
  LIM=""; LEVELS="0 1 2 3 4 5 6 7 8"; SW="--switches 60"; PH="15W:50,7W:50,15W:40"; LONG="--long-limit 5"
fi
DENSE=()
for spec in "conventional=$M/qwen3-0.6b-conventional" "indep-l0=$M/indep-l0" "indep-l1=$M/indep-l1" \
            "indep-full=$M/indep-full" "apollo-0.5b=$M/Apollo-0.5B:apollo_answer:16" \
            "apollo2-0.5b=$M/Apollo2-0.5B:apollo_answer:16" "apollo-1.8b=$M/Apollo-1.8B:apollo_answer:16" \
            "apollo-2b=$M/Apollo-2B:apollo_answer:16" "apollo2-2b=$M/Apollo2-2B:apollo_answer:16" \
            "qwen3-0.6b-reasoning=$M/Qwen3-0.6B:as_is:1024"; do
  dir=${spec#*=}; dir=${dir%%:*}
  [[ -f "$dir/config.json" ]] && DENSE+=(--dense "$spec") || echo "(skipping ${spec%%=*}: $dir not staged)"
done

echo "== 1/6 preflight (15W)"
python3 jetson/check_env.py --power-mode 15W --elastic "$M/mednest"
echo "== 2/6 widths + baselines at 15W"
python3 jetson/bench_widths.py --power-mode 15W --tag 15W --elastic "$M/mednest" --levels $LEVELS $LIM $LONG "${DENSE[@]}"
echo "== 3/6 widths at 7W (latency profile for the trace)"
python3 jetson/bench_widths.py --power-mode 7W --tag 7W --elastic "$M/mednest" --levels $LEVELS $LIM
echo "== 4/6 switch cost (15W)"
python3 jetson/bench_switch.py --power-mode 15W --elastic "$M/mednest" --levels 0 4 8 $SW
echo "== 5/6 budget trace (15W <-> 7W)"
python3 jetson/bench_trace.py --power-mode 15W --elastic "$M/mednest" --levels $LEVELS --phases "$PH" \
  --profile "15W=$MEDNEST_OUT/widths_15W.jsonl" --profile "7W=$MEDNEST_OUT/widths_7W.jsonl" \
  --policies fixed:max fixed:min elastic reload
echo "== 6/6 tables"
python3 jetson/aggregate.py --results "$MEDNEST_OUT"
sudo -n nvpmodel -m 0 >/dev/null 2>&1 || true   # back to 15W
echo "done: $MEDNEST_OUT"
