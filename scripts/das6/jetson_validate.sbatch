#!/usr/bin/env bash
#SBATCH --job-name=flexslm_jetsonval
#SBATCH --partition=defq
#SBATCH --nodelist=node207
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Expected ~20 min; 1 h limit. node207 only (node205/206 train; node208 is kept free for another user).
#SBATCH --time=01:00:00
#SBATCH --output=jobs/%x-%j.out
# End-to-end dry run of the Jetson scripts (jetson/) on a DAS-6 GPU before they go to the board:
# the pilot MedNest export (3 widths) through check_env -> bench_widths (+ dense baselines) ->
# bench_switch -> bench_trace -> aggregate, with small limits. No rails / nvpmodel here: power is
# recorded as unmeasured (None), which is the path a dev machine takes.
#   sbatch scripts/das6/jetson_validate.sbatch [export dir]
EXP=${1:-hf_export/flexllama__sft.pilot.qwen3_0p6b_nested}
source scripts/das6/common.sh gpu
export MEDNEST_OUT=/var/scratch/$USER/flexslm/jetson_validate/$SLURM_JOB_ID
mkdir -p "$MEDNEST_OUT"
COMMON="--cooldown-c 0 --power-source none"
set -x
python jetson/check_env.py $COMMON --elastic "$EXP" --check-prompts 8
python jetson/bench_widths.py $COMMON --idle-s 1 --tag val --elastic "$EXP" --limit 21 --long-limit 1 \
  --dense conventional=hf_export/flexllama__sft.paper.qwen3_0p6b_conventional/level2 \
  --dense apollo-0.5b=FreedomIntelligence/Apollo-0.5B:apollo_answer:16 \
  --dense qwen3-0.6b-reasoning=Qwen/Qwen3-0.6B:as_is:1024
python jetson/bench_widths.py $COMMON --idle-s 1 --tag val_eager --decode eager --elastic "$EXP" --limit 14
python jetson/bench_widths.py $COMMON --idle-s 1 --tag val_materialize --mode materialize --elastic "$EXP" --levels 0 --limit 7
python jetson/bench_switch.py $COMMON --elastic "$EXP" --switches 12 --load-repeats 1
python jetson/bench_trace.py $COMMON --elastic "$EXP" --profile "gpu=$MEDNEST_OUT/widths_val.jsonl" \
  --phases gpu:24 --policies fixed:max fixed:min elastic reload
python jetson/aggregate.py --results "$MEDNEST_OUT" \
  --cluster "mednest:0=results/gen.hf_export-flexllama__sft.pilot.qwen3_0p6b_nested-level0.t32.samples.jsonl" \
  --cluster "mednest:2=results/gen.hf_export-flexllama__sft.pilot.qwen3_0p6b_nested-level2.t32.samples.jsonl" \
  --cluster "conventional=results/gen.hf_export-flexllama__sft.paper.qwen3_0p6b_conventional-level2.t32.samples.jsonl" \
  --cluster "apollo-0.5b=results/gen.FreedomIntelligence-Apollo-0.5B.t16.papollo_answer.samples.jsonl"
ls -la "$MEDNEST_OUT" "$MEDNEST_OUT/derived"
cat "$MEDNEST_OUT/derived/widths.csv" "$MEDNEST_OUT/derived/trace.csv" "$MEDNEST_OUT/derived/switch.csv"
