#!/usr/bin/env bash
# All co-residency cases, one fresh process each (board, venv active):  bash controller/run_coresident.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-$HOME/mednest-results/coresident_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
nvpmodel -q 2>/dev/null | head -2
for c in small big pair pair3 both; do
  echo "=== $c"; timeout 1500 python3 controller/coresident_test.py --case $c --out "$OUT/$c.jsonl"; echo "exit $?"
  sleep 10
done
for c in pair3 both big; do
  echo "=== $c + GPU load"; timeout 1500 python3 controller/coresident_test.py --case $c --load --out "$OUT/$c.load.jsonl"; echo "exit $?"
  sleep 10
done
echo "done: $OUT"
