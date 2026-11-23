#!/usr/bin/env bash
# Co-residency again with the decoder's avoidable memory removed (--lean); one fresh process per case (board, venv active).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-$HOME/mednest-results/coresident_lean}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
nvpmodel -q 2>/dev/null | head -2
for c in pair pair3 both big small; do
  echo "=== $c"; timeout 1500 python3 controller/coresident_test.py --lean --case $c --out "$OUT/$c.jsonl"; echo "exit $?"; sleep 10
done
for c in pair pair3 both; do
  echo "=== $c + GPU load"; timeout 1500 python3 controller/coresident_test.py --lean --case $c --load --out "$OUT/$c.load.jsonl"; echo "exit $?"; sleep 10
done
echo "done: $OUT"
