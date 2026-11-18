#!/usr/bin/env bash
# Stage-1 KD-weight sweep, HydraLLM sampling pilot, teacher screen and inference-compute curve (user, 2026-11-18).
# Run once on the DAS-6 login node from ~/FlexSLM after sync:   bash scripts/das6/queue_2026-11-18.sh
# Job ids are appended to jobs/queue_2026-11-18.ids.
#
# Unit tests (node207) -> 20-step smoke tests of the new objective / sampling (node206) -> pilots.
# Stage-1 pilots (20M tokens, final recipe, CE + lambda KL from the PMC-corrected Qwen3-1.7B, T 2):
#   node205 (after 28347): lambda 0.1;   node206: Hydra sampling at lambda 0.1, lambda 0.3, lambda 0.5.
#   lambda 0 = s1.qwen3_0p6b.pilot.final, whose Stage 2 is 28347 (already queued).
# Stage 2 (M3's Stage 2, node205) of every pilot -> export (node207) -> answer-only eval of the anchors 0 / 4 / 8.
# Eval only (node207): PMC val perplexity of candidate 7-8B teachers (screen); accuracy vs decoding budget of
# the untrained Qwen3-0.6B / 1.7B (FaithMed's reasoning mode, thinking on, 300 questions per test set).
# node208 is NOT used (user, 2026-11-17).
set -eo pipefail
cd ~/FlexSLM
IDS=jobs/queue_2026-11-18.ids
N205="--nodelist=node205"
N206="--nodelist=node206 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=64"
N207="--nodelist=node207"
EVALNODES="--exclude=node205,node208"

sub() { sbatch --parsable "$@"; }
note() { echo "$*" | tee -a "$IDS"; }
note "=== $(date '+%F %T')"

export_eval() {  # <train job id> <config> <levels...>
  local dep=$1 cfg=$2; shift 2
  local dir="hf_export/flexllama__$cfg"
  local e; e=$(sub --partition=defq $N207 --dependency=afterok:$dep scripts/das6/export.sbatch "$cfg")
  local evs=""
  for l in "$@"; do
    evs="$evs $(sub $EVALNODES --dependency=afterok:$e scripts/das6/evalgen.sbatch "$dir/level$l")"
  done
  note "  export $e eval(levels $*)$evs"
}

T=$(sub scripts/das6/pytest.sbatch test/test_kd_sweep.py test/test_continuum.py test/test_level_weights.py \
    test/test_memopt_equivalence.py)
note "unit tests: $T"
P=s1.qwen3_0p6b.pilot.final
SM1=$(sub $N206 --time=01:00:00 --dependency=afterok:$T --export=ALL,SMOKE_STEPS=20 scripts/das6/train.sbatch $P.kdadd0p1.corr)
SM2=$(sub $N206 --time=01:00:00 --dependency=afterok:$SM1 --export=ALL,SMOKE_STEPS=20 scripts/das6/train.sbatch $P.kdadd0p1.corr.hydra)
note "smoke: $SM1 $SM2"

# Stage 1 (node205 ~5 h, node206 ~9 h; Hydra ~4 h on node206: one width per step)
L1=$(sub $N205 --time=12:00:00 --dependency=afterok:$SM2 scripts/das6/train.sbatch $P.kdadd0p1.corr)
H1=$(sub $N206 --time=10:00:00 --dependency=afterok:$SM2 scripts/das6/train.sbatch $P.kdadd0p1.corr.hydra)
L3=$(sub $N206 --time=20:00:00 --dependency=afterok:$SM2 scripts/das6/train.sbatch $P.kdadd0p3.corr)
L5=$(sub $N206 --time=20:00:00 --dependency=afterok:$SM2 scripts/das6/train.sbatch $P.kdadd0p5.corr)
note "Stage 1: lambda0.1 $L1 (node205)  hydra lambda0.1 $H1  lambda0.3 $L3  lambda0.5 $L5 (node206)"
for pair in "$L1:$P.kdadd0p1.corr" "$H1:$P.kdadd0p1.corr.hydra" "$L3:$P.kdadd0p3.corr" "$L5:$P.kdadd0p5.corr"; do
  j=${pair%%:*}; cfg=${pair#*:}
  note "  continuum $cfg: $(sub $EVALNODES --dependency=afterok:$j scripts/das6/eval_continuum.sbatch $cfg --no-mixnmatch)"
done

# Stage 2 (node205, ~5.5 h each), export, eval of the anchors
for pair in "$L1:kdadd0p1.corr" "$H1:kdadd0p1.corr.hydra" "$L3:kdadd0p3.corr" "$L5:kdadd0p5.corr"; do
  j=${pair%%:*}; tag=${pair#*:}
  cfg=sft.pilot.qwen3_0p6b_final9.$tag
  s2=$(sub $N205 --time=12:00:00 --dependency=afterok:$j scripts/das6/train.sbatch $cfg)
  note "Stage 2 $cfg: $s2"
  export_eval "$s2" "$cfg" 0 4 8
done

# Teacher screen: PMC val perplexity (same 640 x 1,024 windows; Qwen3-1.7B 7.92, II-Medical-8B 7.46, corrected 5.57)
for m in FreedomIntelligence/Apollo2-7B FreedomIntelligence/HuatuoGPT-o1-7B Qwen/Qwen3-8B; do
  note "pmcppl $m: $(sub $N207 --time=02:00:00 scripts/das6/eval_continuum.sbatch --hf $m --batch-size 2)"
done

# NOTE: sbatch --export splits at commas, so BUDGETS=64,128,... must come from the submitting environment (env VAR=..).
# (As first submitted, 28396 got BUDGETS=64 only: its curve is rescored offline with eval/rescore_budgets.py.)
# Inference-compute curve: untrained Qwen3, FaithMed's reasoning mode (thinking on, Qwen3 sampling), 300 per set
B=64,128,256,512,1024,2048,4096
ES=$(env PROMPT_FORMAT=think BUDGETS=64,256 PER_SOURCE=2 SAMPLE=qwen3 sbatch --parsable $N207 --time=00:30:00      scripts/das6/evalgen.sbatch Qwen/Qwen3-0.6B 256 16)
note "curve smoke (2 per set, 256 tokens): $ES"
note "curve 1.7B think: $(env PROMPT_FORMAT=think BUDGETS=$B PER_SOURCE=300 SAMPLE=qwen3 sbatch --parsable $N207 --time=14:00:00 --dependency=afterok:$ES scripts/das6/evalgen.sbatch Qwen/Qwen3-1.7B 4096 16)"
note "curve 0.6B think: $(env PROMPT_FORMAT=think BUDGETS=$B PER_SOURCE=300 SAMPLE=qwen3 sbatch --parsable $N207 --time=12:00:00 --dependency=afterok:$ES scripts/das6/evalgen.sbatch Qwen/Qwen3-0.6B 4096 32)"
