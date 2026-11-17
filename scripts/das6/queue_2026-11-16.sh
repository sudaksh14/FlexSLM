#!/usr/bin/env bash
# Final-recipe queue (2026-11-16). Run once on the DAS-6 login node from ~/FlexSLM after sync:
#   bash scripts/das6/queue_2026-11-16.sh
# Job ids are appended to jobs/queue_2026-11-16.ids.
#
# node205 (2 x A40), in this order:
#   Stage 2 on the best 20M pilot (KD, then CE only) -> A2 no Stage 1 -> B2 MatFormer-style
#   -> 1.7B Stage-1 memory smoke -> final 100M Stage 1 (0.6B) -> final Stage 2 (KD, CE only)
#   -> Stage 2 of the independent baselines -> 1.7B Stage 1 (100M)
# node206 (1 x A40; Stage 1 uses a global batch, so 1 GPU takes the same steps):
#   final recipe at 20M (gate of the 100M run) -> independent L0 -> independent L1 -> no ordering
# node207: untrained Qwen3-0.6B with 1,024 new tokens (B0), Minitron ordering of Qwen3-1.7B
# Exports: fatq is fully occupied (another user's 15-day jobs on node201-203), so the CPU export
# jobs run on node208 (defq, no GPU requested), a few minutes each.
set -eo pipefail
cd ~/FlexSLM
IDS=jobs/queue_2026-11-16.ids
N205="--nodelist=node205"
N206="--nodelist=node206 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=64"
EVALNODES="--exclude=node205,node206"

sub() { sbatch --parsable "$@"; }
note() { echo "$*" | tee -a "$IDS"; }

# export + answer-only evaluation (32 tokens) of the given levels, after training job $1
export_eval() {  # <train job id> <config> <levels...>
  local dep=$1 cfg=$2; shift 2
  local dir="hf_export/flexllama__$cfg"
  local e; e=$(sub --partition=defq --nodelist=node208 --dependency=afterok:$dep scripts/das6/export.sbatch "$cfg")
  local evs=""
  for l in "$@"; do
    evs="$evs $(sub $EVALNODES --dependency=afterok:$e scripts/das6/evalgen.sbatch "$dir/level$l")"
  done
  note "  export $e eval(levels $*)$evs"
}

# ---- node207: B0 and the 1.7B ordering
B0=$(sub --nodelist=node207 --time=16:00:00 scripts/das6/evalgen.sbatch Qwen/Qwen3-0.6B 1024 32)
note "B0 untrained Qwen3-0.6B, 1,024 new tokens: $B0"
ORD17=$(sub --nodelist=node208 scripts/das6/importance_order.sbatch --model Qwen/Qwen3-1.7B --scheme minitron \
        --out hf_export/qwen3-1.7b.minitron --eval-seqs 0)
note "Minitron ordering of Qwen3-1.7B: $ORD17"

# ---- node206: final recipe at 20M, then the 100M single-GPU Stage-1 runs
PF=$(sub $N206 --time=16:00:00 scripts/das6/train.sbatch s1.qwen3_0p6b.pilot.final)
PFC=$(sub $EVALNODES --dependency=afterok:$PF scripts/das6/eval_continuum.sbatch s1.qwen3_0p6b.pilot.final)
note "pilot final (20M, node206): $PF continuum $PFC"
IL0=$(sub $N206 --time=40:00:00 --dependency=afterok:$PF scripts/das6/train.sbatch s1.qwen3_0p6b.full.indep_l0)
IL1=$(sub $N206 --time=40:00:00 --dependency=afterany:$IL0 scripts/das6/train.sbatch s1.qwen3_0p6b.full.indep_l1)
NOORD=$(sub $N206 --time=90:00:00 --dependency=afterany:$IL1 scripts/das6/train.sbatch s1.qwen3_0p6b.full.no_order)
note "Stage 1 100M on node206: indep_l0 $IL0 indep_l1 $IL1 no_order $NOORD"

# ---- node205: Stage 2 on the best 20M pilot, A2, B2
S2P=$(sub $N205 scripts/das6/train.sbatch sft.pilot.qwen3_0p6b_nested)
note "Stage 2 on pilot order_minitron_lr2e-4 (KD): $S2P"; export_eval $S2P sft.pilot.qwen3_0p6b_nested 0 1 2
S2PN=$(sub $N205 --dependency=afterany:$S2P scripts/das6/train.sbatch sft.pilot.qwen3_0p6b_nested.no_kd)
note "Stage 2 on pilot (CE only): $S2PN"; export_eval $S2PN sft.pilot.qwen3_0p6b_nested.no_kd 0 1 2
A2=$(sub $N205 --dependency=afterany:$S2PN scripts/das6/train.sbatch sft.paper.qwen3_0p6b_nested.no_stage1)
note "A2 no Stage 1: $A2"; export_eval $A2 sft.paper.qwen3_0p6b_nested.no_stage1 0 1 2
B2=$(sub $N205 --dependency=afterany:$A2 scripts/das6/train.sbatch sft.paper.qwen3_0p6b_matformer)
note "B2 MatFormer-style: $B2"; export_eval $B2 sft.paper.qwen3_0p6b_matformer 0 1 2

# ---- node205: 1.7B Stage-1 memory smoke (20 steps, longest batches)
SM17=$(sub $N205 --time=01:30:00 --export=ALL,SMOKE_STEPS=20 --dependency=afterok:$ORD17,afterany:$B2 \
       scripts/das6/train.sbatch s1.dev.qwen3_1p7b_smoke)
note "1.7B Stage-1 smoke: $SM17"

# ---- node205: final 100M Stage 1 (after the 20M run of the same recipe succeeded), final Stage 2
FULL=$(sub $N205 --time=44:00:00 --dependency=afterok:$PF,afterany:$SM17 scripts/das6/train.sbatch s1.qwen3_0p6b.full)
FULLC=$(sub $EVALNODES --dependency=afterok:$FULL scripts/das6/eval_continuum.sbatch s1.qwen3_0p6b.full)
note "FINAL Stage 1 (100M): $FULL continuum $FULLC"
M3=$(sub $N205 --dependency=afterok:$FULL scripts/das6/train.sbatch sft.paper.qwen3_0p6b_nested)
note "M3 final Stage 2 (KD): $M3"; export_eval $M3 sft.paper.qwen3_0p6b_nested 0 1 2 3 4 5 6 7 8
M3N=$(sub $N205 --dependency=afterok:$FULL,afterany:$M3 scripts/das6/train.sbatch sft.paper.qwen3_0p6b_nested.no_kd)
note "M3 final Stage 2 (CE only): $M3N"; export_eval $M3N sft.paper.qwen3_0p6b_nested.no_kd 0 1 2 3 4 5 6 7 8

# ---- node205: Stage 2 of the independent baselines and of the no-ordering ablation
I0=$(sub $N205 --dependency=afterok:$IL0,afterany:$M3N scripts/das6/train.sbatch sft.paper.qwen3_0p6b_indep_l0)
note "I1 independent 0.5 width, Stage 2: $I0"; export_eval $I0 sft.paper.qwen3_0p6b_indep_l0 0
I1=$(sub $N205 --dependency=afterok:$IL1,afterany:$I0 scripts/das6/train.sbatch sft.paper.qwen3_0p6b_indep_l1)
note "I1 independent 0.75 width, Stage 2: $I1"; export_eval $I1 sft.paper.qwen3_0p6b_indep_l1 4
A1=$(sub $N205 --dependency=afterok:$NOORD scripts/das6/train.sbatch sft.paper.qwen3_0p6b_nested.no_order)
note "A1 no ordering, Stage 2: $A1"; export_eval $A1 sft.paper.qwen3_0p6b_nested.no_order 0 4 8

# ---- node205: 1.7B Stage 1 (100M tokens, CE only), after the 0.6B Stage-2 runs
S17=$(sub $N205 --time=140:00:00 --dependency=afterok:$SM17,afterany:$M3N scripts/das6/train.sbatch s1.qwen3_1p7b.full)
note "1.7B Stage 1 (100M): $S17"
squeue -u "$USER" -o "%.8i %.28j %.3t %.10M %.20R %E" | head -80
