#!/usr/bin/env bash
# Sourced by every FlexSLM job on hipster (UvA FNWI cluster). Hipster conventions, taken from
# the FlexLLaVA hipster launchers (Collaborative VLM/FlexLLaVA, hipster branch):
#  - partitions: performance = 5 nodes x 8 RTX 6000 Ada 48 GB, 32 CPUs per GPU enforced;
#    capacity = 8 nodes x 8 L4 24 GB, 16 CPUs per GPU; max 7 days; shared multi-tenant
#    cluster, so never --exclusive, request exactly the GPUs/CPUs needed;
#  - CUDA module cuda/12.9.1; conda env my_env (torch 2.8.0, transformers 4.57.6, lightning
#    2.5.2, python 3.9), used read-only, nothing is installed into it;
#  - storage (global rules: ~/.claude/CLAUDE.md): /scratch is wiped every few weeks, so it holds ONLY
#    checkpoints and reproducible items (store/, hf_export/, jobs/); repo root, data, results and
#    caches live in /home (200 GB, shared by all projects). Until scripts/hipster/migrate_to_home.sbatch
#    has run, data/results/HF cache are still in /scratch/$USER/flexslm (symlinked into the repo).
# The code lives in ~/FlexSLM-medbench (the older ~/FlexSLM clone on hipster is left alone).
# Usage at the top of a job script:  source scripts/hipster/common.sh gpu|cpu
set -eo pipefail

KIND=${1:-gpu}
export FLEXSLM_SCRATCH=/scratch/$USER/flexslm
export FLEXSLM_STORE=${FLEXSLM_STORE:-$FLEXSLM_SCRATCH/store}
# Hugging Face cache: the home copy once it exists (after the migration), else the scratch one.
if [[ -d /home/$USER/flexslm_saves/cache/huggingface ]]; then
  export HF_HOME=/home/$USER/flexslm_saves/cache/huggingface
else
  export HF_HOME=$FLEXSLM_SCRATCH/hf_cache
fi
export HF_TOKEN=$(cat "$HOME/.hf_token" 2>/dev/null || true)
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_TIMEOUT=3600 NCCL_DEBUG=WARN NCCL_ASYNC_ERROR_HANDLING=1

if [[ "$KIND" == gpu ]]; then
  if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
  module load cuda/12.9.1 2>&1 || echo "[common] WARNING: module load cuda/12.9.1 failed (torch ships its own runtime)"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate my_env

cd "${SLURM_SUBMIT_DIR:-$HOME/FlexSLM-medbench}"
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
