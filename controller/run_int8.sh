#!/usr/bin/env bash
# int8 vs bf16, nested 1.7B vs the pair (board, venv active). One fresh process per case; energy only without load.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-$HOME/mednest-results/int8}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
nvpmodel -q 2>/dev/null | head -2
for c in nested; do echo "=== bf16 $c"; timeout 2400 python3 controller/int8_test.py --case $c --prec bf16 --out "$OUT/$c.bf16.jsonl"; echo "exit $?"; sleep 10; done
for c in small big nested pair pairn both; do
  echo "=== int8 $c"; timeout 2400 python3 controller/int8_test.py --case $c --prec int8 --n 4 --no-energy --out "$OUT/$c.int8.jsonl"; echo "exit $?"; sleep 10
done
for c in nested pair both; do
  echo "=== int8 $c + GPU load"; timeout 2400 python3 controller/int8_test.py --case $c --prec int8 --n 4 --load --out "$OUT/$c.int8.load.jsonl"; echo "exit $?"; sleep 10
done
echo "done: $OUT"
