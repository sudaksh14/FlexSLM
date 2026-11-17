#!/usr/bin/env bash
# Put every model the Jetson runs need under $MODELS (default ~/mednest/models). Run on the board
# (or on any machine that reaches DAS-6 and the Hugging Face Hub, then copy $MODELS over).
#
#   bash jetson/stage_models.sh            # MedNest final + baselines
#   ONLY="mednest" bash jetson/stage_models.sh
#
# From DAS-6 (rsync over ssh; needs your key for $DAS6), exports in hf_export/:
#   mednest       final nested model: level8 (full weights, served elastically) + every level's
#                 config/tokenizer + the dense anchor levels 0 and 4 (bench_switch path C and
#                 check_env's view-vs-export proof). ALL_LEVELS=1 copies every dense level (6.8 GB).
#   conventional  conventional fine-tuning of Qwen3-0.6B (paper baseline, same size as level8)
#   indep-*       independent 0.5 / 0.75 / full-width models (once trained)
#   apollo-1.8b   the dot-free local snapshot used on the cluster (Qwen-1 code, reviewed, commit 73072ac)
# From the Hub, pinned to the commits evaluated on the cluster (2026-11-17):
#   Qwen3-0.6B (untrained, reasons with 1,024 tokens), Apollo-0.5B, Apollo2-0.5B, Apollo-2B, Apollo2-2B.
set -euo pipefail
MODELS=${MODELS:-$HOME/mednest/models}
DAS6=${DAS6:-skalra@fs2.das6.science.uva.nl}
EXPORT=${EXPORT:-FlexSLM/hf_export}                     # relative to the DAS-6 home directory
FINAL=${FINAL:-flexllama__sft.paper.qwen3_0p6b_nested}  # MedNest final export (KD); .no_kd if CE-only wins
ONLY=${ONLY:-mednest conventional indep apollo hub}
mkdir -p "$MODELS"
want() { [[ " $ONLY " == *" $1 "* ]]; }
pull() {  # <remote dir under $EXPORT> <local name> [rsync filters...]
  local src=$1 dst=$2; shift 2
  echo "== $src -> $MODELS/$dst"
  rsync -a --info=progress2 "$@" "$DAS6:$EXPORT/$src/" "$MODELS/$dst/"
}

if want mednest; then
  if [[ "${ALL_LEVELS:-0}" == 1 ]]; then
    pull "$FINAL" mednest
  else
    # weights of level8 (served) and of the anchors 0 and 4 (dense comparisons); config + tokenizer of all
    pull "$FINAL" mednest --include='level8/***' --include='level0/***' --include='level4/***' \
         --include='level*/' --include='level*/*.json' --include='level*/*.txt' --exclude='*'
  fi
fi
if want conventional; then pull flexllama__sft.paper.qwen3_0p6b_conventional/level2 qwen3-0.6b-conventional; fi
if want indep; then
  for n in l0 l1 full; do
    lvl=$([[ $n == l0 ]] && echo 0 || ([[ $n == l1 ]] && echo 4 || echo 8))
    ssh -o BatchMode=yes "$DAS6" "test -d $EXPORT/flexllama__sft.paper.qwen3_0p6b_indep_$n/level$lvl" \
      && pull "flexllama__sft.paper.qwen3_0p6b_indep_$n/level$lvl" "indep-$n" \
      || echo "   (indep_$n not exported yet: skipped)"
  done
fi
if want apollo; then pull apollo_1p8b Apollo-1.8B; fi
if want hub; then
  python3 - "$MODELS" <<'EOF'
import sys
from huggingface_hub import snapshot_download
pins = {  # repo: commit evaluated on the cluster
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "FreedomIntelligence/Apollo-0.5B": "1a57c382aa040f76e22a40fdf2953030fcdff7f5",
    "FreedomIntelligence/Apollo2-0.5B": "044ffc6a510bc1fb2f7a52d5f33dd861fa718118",
    "FreedomIntelligence/Apollo-2B": "4049fc07e95649c91e86be4458faf935bdc106b8",
    "FreedomIntelligence/Apollo2-2B": "ee3b5f008ba59fa9662f746a942de3dace2112e0",
}
for repo, rev in pins.items():
    d = f"{sys.argv[1]}/{repo.split('/')[1]}"
    print("==", repo, rev, "->", d, flush=True)
    snapshot_download(repo, revision=rev, local_dir=d,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.tiktoken"])
EOF
fi
du -sh "$MODELS"/* 2>/dev/null || true
