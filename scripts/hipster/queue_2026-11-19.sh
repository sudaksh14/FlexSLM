#!/usr/bin/env bash
# Single-run (no Stage 1) ablations, independent baselines and the 1.7B MedNest (user, 2026-11-18), hipster.
# Run once on the hipster login node from ~/FlexSLM-medbench after sync:  bash scripts/hipster/queue_2026-11-18.sh
# Job ids are appended to jobs/queue_2026-11-18.hipster.ids (and recorded in implementation_log.md).
#  0. migrate_to_home (capacity): data, results, jobs/, HF cache out of /scratch (wiped every few weeks);
#     SKIP_JOB_CHECK because every other job here waits for it (afterok), none can run meanwhile.
#  1. fetch the exports the runs need from DAS-6 (fine-tuned 1.7B teacher, Minitron-ordered 0.6B / 1.7B).
#  2. unit tests (capacity CPU) -> 20-step smoke tests (0.6B one-width sampling, 1.7B memory), capacity L4.
#  3. seven trainings on capacity (0.6B: 2 x L4, 16 per GPU; 1.7B: 4 x L4, 8 per GPU; effective batch 32 as on DAS-6),
#     each -> CPU export (capacity) -> 32-token evals (capacity, 1 L4) of the trained levels.
set -eo pipefail
cd ~/FlexSLM-medbench
IDS=queue_2026-11-19b.hipster.ids   # repo root until jobs/ is a real directory (moved by the migration)
# capacity (L4 24 GB, 16 CPUs per GPU): performance was estimated to start only on 2026-12-04 (sbatch --test-only).
# performance (RTX 6000 Ada 48 GB, 32 CPUs per GPU enforced): 2-GPU trainings that need > 24 GB (hipster rules). The first
# smoke on capacity L4s ran out of memory (398151). 0.6B and 1.7B both: 16 per GPU x 2 = 32 per step, as on DAS-6.
C2="--partition=performance --gres=gpu:rtx_6000_ada:2 --ntasks-per-node=2 --cpus-per-task=32"
C4="$C2"
sub() { sbatch --parsable "$@"; }
note() { echo "$*" | tee -a "$IDS"; }
note "=== $(date '+%F %T')"

# 2026-11-19 recovery: migrate (397255) and fetch (397256) finished; the 2026-11-18 chain died on an import error in
# training/sft.py (fixed). Everything still pending was cancelled first.
for n in flexslm_train flexslm_export flexslm_evalgen flexslm_pytest; do scancel -u "$USER" -n $n || true; done
note "cancelled every flexslm_{train,export,evalgen,pytest} job (old chain + the hung smoke 398151)"
T=$(sub scripts/hipster/pytest.sbatch test/test_kd_sweep.py test/test_continuum.py test/test_level_weights.py)
note "tests $T"
A=sft.paper.qwen3_0p6b_nested.no_stage1
SM1=$(sub $C2 --time=01:00:00 --dependency=afterok:$T --export=ALL,SMOKE_STEPS=20 scripts/hipster/train.sbatch $A.single)
SM2=$(sub $C4 --time=01:00:00 --dependency=afterok:$T --export=ALL,SMOKE_STEPS=20 scripts/hipster/train.sbatch sft.paper.qwen3_1p7b_nested.no_stage1)
note "smoke $SM1 $SM2"

run() {  # <gpus flags var> <config> <time> <levels...>
  local res=$1 cfg=$2 t=$3; shift 3
  local j e evs=""
  j=$(sub ${!res} --time=$t --dependency=afterok:$SM1:$SM2 scripts/hipster/train.sbatch "$cfg")
  e=$(sub --dependency=afterok:$j scripts/hipster/export.sbatch "$cfg")
  for l in "$@"; do evs="$evs $(sub --dependency=afterok:$e scripts/hipster/evalgen.sbatch "hf_export/flexllama__$cfg/level$l")"; done
  note "$cfg: train $j export $e eval(levels $*)$evs"
}
# A2 took 4.3 h on 2 x A40 (3 levels); an L4 is slower (~1.3-1.5x), one level ~1/2 of the time;
# 1.7B, 3 levels, CE, ZeRO-1 + checkpointing on 4 x L4 ~10 h. --time = 2x.
run C2 $A.no_kd    10:00:00 0 1 2
run C2 $A.no_order 10:00:00 0 1 2
run C2 $A.single   08:00:00 0 1 2
run C2 $A.indep_l0 06:00:00 0
run C2 $A.indep_l1 06:00:00 1
run C2 $A.indep_l2 08:00:00 2
run C4 sft.paper.qwen3_1p7b_nested.no_stage1 16:00:00 0 1 2
