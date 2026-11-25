#!/usr/bin/env bash
# The int8 cases that did not run on 2026-11-23 (stopped early). Board, venv active. Output kept apart from the first run.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-$HOME/mednest-results/int8_resume}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
nvpmodel -q 2>/dev/null | head -2
for c in nested pair pairn both; do
  echo "=== int8 $c"; timeout 2400 python3 controller/int8_test.py --case $c --prec int8 --n 4 --no-energy --out "$OUT/$c.int8.jsonl"; echo "exit $?"; sleep 10
done
for c in nested pair both; do
  echo "=== int8 $c + GPU load"; timeout 2400 python3 controller/int8_test.py --case $c --prec int8 --n 4 --load --out "$OUT/$c.int8.load.jsonl"; echo "exit $?"; sleep 10
done
echo "done: $OUT"
