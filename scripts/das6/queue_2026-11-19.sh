#!/usr/bin/env bash
# Recovery queue (2026-11-19). The jobs of queue_2026-11-18.sh that started after the sync of the 2026-11-18 evening
# died on import (training/sft.py: `no_order` ablation passed hf_start twice; fixed, 52 tests pass, job 28406).
# What had finished: 28347 (Stage 2 of the 20M final-recipe pilot), 28365 (Stage 1, Hydra sampling, lambda 0.1).
# Cancelled and resubmitted here: exports/evals of 28347, the Hydra continuum + Stage 2, Stage 1 for lambda 0.1 / 0.3 / 0.5
# and their continuum / Stage 2 / export / evals. Run on the DAS-6 login node:  ssh das6 'bash -s' < this file.
# BUDGETS must come from the environment (sbatch --export splits at commas).
set -eo pipefail
cd ~/FlexSLM
IDS=jobs/queue_2026-11-19.ids
N205="--nodelist=node205"
N206="--nodelist=node206 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=64"
N207="--nodelist=node207"
EVALNODES="--exclude=node205,node208"
sub() { sbatch --parsable "$@"; }
note() { echo "$*" | tee -a "$IDS"; }
note "=== $(date '+%F %T')"

scancel $(seq 28349 28357) 28368 28370 28371 $(seq 28372 28376) $(seq 28378 28391) 2>/dev/null || true
note "cancelled: 28349-57 28368 28370 28371 28372-76 28378-91 (broken dependency chains)"

export_eval() {  # <train job id or ""> <config> <levels...>
  local dep=$1 cfg=$2; shift 2
  local dir="hf_export/flexllama__$cfg"
  local e; e=$(sub --partition=defq $N207 ${dep:+--dependency=afterok:$dep} scripts/das6/export.sbatch "$cfg")
  local evs=""
  for l in "$@"; do evs="$evs $(sub $EVALNODES --dependency=afterok:$e scripts/das6/evalgen.sbatch "$dir/level$l")"; done
  note "  export $e eval(levels $*)$evs"
}
S2=sft.pilot.qwen3_0p6b_final9
P=s1.qwen3_0p6b.pilot.final

note "28347 (final-recipe pilot Stage 2, finished): export + evals"
export_eval "" $S2 0 1 2 3 4 5 6 7 8

note "Hydra lambda 0.1 (Stage 1 = 28365, finished): continuum, Stage 2, export, evals"
note "  continuum: $(sub $EVALNODES scripts/das6/eval_continuum.sbatch $P.kdadd0p1.corr.hydra --no-mixnmatch)"
SH=$(sub $N205 --time=12:00:00 scripts/das6/train.sbatch $S2.kdadd0p1.corr.hydra)
note "  Stage 2: $SH"
export_eval "$SH" $S2.kdadd0p1.corr.hydra 0 4 8

for lam in 0p1 0p3 0p5; do
  if [[ $lam == 0p1 ]]; then NODE="$N205"; T=12:00:00; else NODE="$N206"; T=20:00:00; fi
  L=$(sub $NODE --time=$T scripts/das6/train.sbatch $P.kdadd$lam.corr)
  note "Stage 1 lambda ${lam}: $L  continuum $(sub $EVALNODES --dependency=afterok:$L scripts/das6/eval_continuum.sbatch $P.kdadd$lam.corr --no-mixnmatch)"
  S=$(sub $N205 --time=12:00:00 --dependency=afterok:$L scripts/das6/train.sbatch $S2.kdadd$lam.corr)
  note "  Stage 2: $S"
  export_eval "$S" $S2.kdadd$lam.corr 0 4 8
done

# 1.7B reasoning curve (28396 finished with BUDGETS=64 only): rescore the stored generations at every budget (CPU).
note "rescore 28396: $(sbatch --parsable --partition=defq $N207 --time=00:40:00 --output=jobs/%x-%j.out --job-name=flexslm_rescore \
   --wrap 'cd ~/FlexSLM; source scripts/das6/common.sh cpu; python -m eval.rescore_budgets results/gen.Qwen-Qwen3-1.7B.t4096.pthink.n300.sq3 --budgets 64,128,256,512,1024,2048,4096 --tokenizer Qwen/Qwen3-1.7B --per_source 300')"
