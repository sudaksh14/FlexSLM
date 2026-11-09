#!/usr/bin/env bash
# Sourced by every FlexSLM job on DAS-6. Encodes the DAS-6 rules:
#  - GPU jobs load the CUDA toolkit module (without it torch silently runs on CPU)
#    and check that torch sees the GPU before doing any work;
#  - the conda prefix env /var/scratch/$USER/prune_llm (torch 2.7.0+cu126,
#    lightning 2.5.3) is activated, plus the FlexSLM pylibs overlay
#    (transformers 4.51.1 for Qwen3, tokenizers, pytest); nothing is installed
#    into prune_llm itself;
#  - heavy files live under /var/scratch/$USER/flexslm, code in ~/FlexSLM.
# Usage at the top of a job script:  source scripts/das6/common.sh gpu|cpu
set -eo pipefail

KIND=${1:-gpu}
export FLEXSLM_SCRATCH=/var/scratch/$USER/flexslm
export FLEXSLM_STORE=${FLEXSLM_STORE:-$FLEXSLM_SCRATCH/store}   # checkpoints, saved_models, wandb
export HF_HOME=$FLEXSLM_SCRATCH/hf_cache
export HF_TOKEN=$(cat "$HOME/.hf_token" 2>/dev/null || true)
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "$KIND" == gpu ]]; then
  if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
  module load cuda12.6/toolkit/12.6
fi

source "/var/scratch/$USER/minimonda3/etc/profile.d/conda.sh"
conda activate "/var/scratch/$USER/prune_llm"
export PYTHONPATH="$FLEXSLM_SCRATCH/pylibs${PYTHONPATH:+:$PYTHONPATH}"

cd "${SLURM_SUBMIT_DIR:-$HOME/FlexSLM}"
echo "[common] host=$(hostname) job=${SLURM_JOB_ID:-none} kind=$KIND python=$(which python)"
python -c "
import torch, transformers
print(f'[common] torch {torch.__version__} transformers {transformers.__version__} '
      f'cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'[common]   cuda:{i} {torch.cuda.get_device_name(i)}')
"
if [[ "$KIND" == gpu ]]; then
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "[common] ERROR: GPU job but torch sees no GPU" >&2; exit 4; }
fi
