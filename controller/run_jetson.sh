#!/usr/bin/env bash
# Width controller on the Jetson Orin Nano (run on the board, inside tmux, venv active).
#   . ~/mednest-venv/bin/activate && cd ~/FlexSLM
#   QUICK=1 bash controller/run_jetson.sh       # smoke: 42 requests, 3 policies (~5 min)
#   bash controller/run_jetson.sh               # full: 630 requests x 6 policies (~35 min, estimate)
#   PHASES=15W:210,7W:210,15W:210 bash controller/run_jetson.sh   # power-mode disturbance instead of load
#   NO_DEADLINES=1 bash controller/run_jetson.sh                  # budget-only run (queues vs fixed)
# Default model: the random-weight shape-only export already on the board (no checkpoint needed);
# correctness and answer lengths come from the cluster (--score cluster). With the trained export
# staged (jetson/stage_models.sh): ELASTIC=~/mednest/models/mednest SCORE=device bash controller/run_jetson.sh
# Copy $MEDNEST_OUT back, then on the laptop: python controller/analyze_board.py <dir>/controller_*.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/mednest/models}
ELASTIC=${ELASTIC:-$M/mednest-random}
SCORE=${SCORE:-cluster}
PROFILE=${PROFILE:-controller/bundles/profile.0p6b.15W.jsonl}
export MEDNEST_OUT=${MEDNEST_OUT:-$HOME/mednest-results/controller_$(date +%Y%m%d_%H%M)}
mkdir -p "$MEDNEST_OUT"
exec > >(tee -a "$MEDNEST_OUT/run_controller.log") 2>&1

if [[ "${QUICK:-0}" == 1 ]]; then
  BUNDLE=controller/bundles/smoke.json
  PHASES=${PHASES:-15W:14,15W+gpu:14,15W:14}
  POLICIES="fixed:max deadline+fb queue-task:0.5"
  TAG=smoke
else
  BUNDLE=${BUNDLE:-controller/bundles/paper.qwen3_0p6b_nested.n630.s0.json}
  PHASES=${PHASES:-15W:210,15W+gpu:210,15W:210}
  POLICIES=${POLICIES:-fixed:max fixed:min deadline deadline+fb queue:0.5 queue-task:0.5}
  TAG=${TAG:-full}
fi
EXTRA=()
[[ "${NO_DEADLINES:-0}" == 1 ]] && EXTRA+=(--no-deadlines) && TAG=${TAG}_nodl

# Without an export on disk (the board's disk is small): random weights built in memory from the base model.
if [[ -f "$ELASTIC/level8/config.json" ]]; then
  MODEL=(--elastic "$ELASTIC")
elif [[ "$SCORE" == cluster && -f "$M/Qwen3-0.6B/config.json" ]]; then
  echo "no export at $ELASTIC: random weights in memory from $M/Qwen3-0.6B"
  MODEL=(--random-from "$M/Qwen3-0.6B")
else
  echo "no export at $ELASTIC (stage_models.sh for --score device)"; exit 2
fi
python3 controller/test_controller.py
python3 controller/jetson_runner.py --power-mode 15W --tag "$TAG" --bundle "$BUNDLE" "${MODEL[@]}" \
  --profile "$PROFILE" --phases "$PHASES" --policies $POLICIES --score "$SCORE" "${EXTRA[@]}"
sudo -n nvpmodel -m 0 >/dev/null 2>&1 || true   # back to the default mode
echo "done: $MEDNEST_OUT"
