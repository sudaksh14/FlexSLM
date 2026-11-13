#!/usr/bin/env bash
# Jobs prepared while DAS-6 was down on 2026-11-12 (implementation_log.md). Run ONCE on the
# login node from ~/FlexSLM, after syncing:
#   SSH_AUTH_SOCK=$HOME/.ssh/agent.sock bash scripts/das6/sync.sh     (local)
#   bash scripts/das6/queue_2026-11-12.sh                             (DAS-6)
# Check first whether 28150-28155 (B1 chain, round-2 pilots) survived the outage
# (squeue / jobs/*.out); this script does not depend on them.
set -euo pipefail
cd ~/FlexSLM
sb() { sbatch --parsable "$@"; }
log() { echo "$*" | tee -a jobs/queue_2026-11-12.ids; }
P=s1.qwen3_0p6b.pilot
N206=(--nodelist=node206 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=64)  # all 64 CPUs: DAS-6 rule

# --- 1. No training: teacher / student PMC perplexity, continuum of existing pilots -------
log "pmcppl Qwen3-1.7B $(sb scripts/das6/eval_continuum.sbatch --hf Qwen/Qwen3-1.7B)"
log "pmcppl Qwen3-0.6B $(sb scripts/das6/eval_continuum.sbatch --hf Qwen/Qwen3-0.6B)"
log "continuum base.init $(sb scripts/das6/eval_continuum.sbatch $P.base --init)"
for arm in base levels5 levels5_sw3 grid9_sw3 order_taylor lr2e-4; do
  log "continuum $arm $(sb scripts/das6/eval_continuum.sbatch $P.$arm)"
done

# --- 2. Minitron ordering checkpoint + scoring cost (node207, in sequence) ----------------
o=$(sb --nodelist=node207 scripts/das6/importance_order.sbatch --scheme minitron --out hf_export/qwen3-0.6b.minitron)
t=$(sb --nodelist=node207 --dependency=afterany:$o scripts/das6/importance_order.sbatch --scheme taylor \
      --out /var/scratch/$USER/flexslm/timing/taylor --eval-seqs 0)
m=$(sb --nodelist=node207 --dependency=afterany:$t scripts/das6/importance_order.sbatch --scheme minitron \
      --batch-size 8 --out /var/scratch/$USER/flexslm/timing/minitron_b8 --eval-seqs 0)
log "order minitron $o timing taylor $t minitron_b8 $m"

# --- 3. Training: node205 (2 GPUs, ~2.7 h per 20M pilot) ----------------------------------
tc=$(sb --time=10:00:00 scripts/das6/train.sbatch s1.qwen3_1p7b.teacher_correction)
tx=$(sb --dependency=afterok:$tc scripts/das6/export.sbatch s1.qwen3_1p7b.teacher_correction)
log "teacher_correction $tc export $tx"
log "pmcppl corrected $(sb --dependency=afterok:$tx scripts/das6/eval_continuum.sbatch --hf \
      hf_export/flexllama__s1.qwen3_1p7b.teacher_correction/level0)"
train205() {  # <arm> [dependency]
  local dep=${2:+--dependency=afterok:$2} j
  j=$(sb --time=07:00:00 $dep scripts/das6/train.sbatch $P.$1)
  log "pilot $1 $j continuum $(sb --dependency=afterok:$j scripts/das6/eval_continuum.sbatch $P.$1)"
}
train205 kd0
train205 T1
train205 kd_corrected "$tx"
train205 kd1_corrected "$tx"
train205 order_minitron "$o"
train205 order_minitron_lr2e-4 "$o"

# --- 4. Training: node206 (1 GPU, ~5 h per 20M pilot; continuum methods 2 and 3) ----------
train206() {  # <arm> <time>
  local j
  j=$(sb "${N206[@]}" --time=$2 scripts/das6/train.sbatch $P.$1)
  log "pilot $1 $j continuum $(sb --dependency=afterok:$j scripts/das6/eval_continuum.sbatch $P.$1)"
}
train206 grid9_anchors_plus1 16:00:00
train206 level_norms 12:00:00
train206 mod_scale 12:00:00
train206 mod_scale_shift 12:00:00
echo "job ids appended to jobs/queue_2026-11-12.ids"
