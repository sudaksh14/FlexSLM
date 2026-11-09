#!/usr/bin/env bash
# One-off setup on the DAS-6 login node (network I/O only, no compute):
#  1. the FlexSLM pylibs overlay on top of prune_llm (Qwen3 needs transformers>=4.51);
#  2. scratch directories, and repo symlinks data/ hf_export/ results/ jobs/ -> scratch;
#  3. pre-downloads the Qwen3 checkpoints into the HF cache (jobs only read it).
#   bash scripts/das6/setup_env.sh        (from ~/FlexSLM)
set -eo pipefail
S=/var/scratch/$USER/flexslm
mkdir -p "$S"/{pylibs,hf_cache,store,data,hf_export,results,jobs}
PY=/var/scratch/$USER/prune_llm/bin/python
$PY -m pip install --quiet --no-deps --target "$S/pylibs" --upgrade \
    transformers==4.51.1 tokenizers==0.21.1 \
    pytest==8.3.5 pluggy==1.5.0 iniconfig==2.0.0 exceptiongroup==1.2.2 tomli==2.2.1
# prune_llm ships a flash_attn wheel that needs GLIBC 2.32 (DAS-6 has 2.28), and
# transformers>=4.51 imports flash_attn when it is installed. Shadow it with the
# flash-attn 2.8.3 that the Elastoformer-MOO project compiled on DAS-6 for the same
# prune_llm (py3.10, torch 2.7.0+cu126).
EFA=/var/scratch/$USER/elastoformer_moo/pylibs
cp -rn "$EFA"/flash_attn "$EFA"/flash_attn-2.8.3.dist-info "$EFA"/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so "$S/pylibs"/
[[ -d "$EFA/hopper" ]] && cp -rn "$EFA/hopper" "$S/pylibs"/ || true

cd "$HOME/FlexSLM"
for d in data hf_export results jobs; do
  if [[ ! -L $d ]]; then
    if [[ -d $d ]]; then cp -rn "$d"/. "$S/$d"/ && rm -rf "$d"; fi
    ln -s "$S/$d" "$d"
  fi
done

export HF_HOME=$S/hf_cache PYTHONPATH=$S/pylibs
$PY -c "
from huggingface_hub import snapshot_download
for m in ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B']:
    print('cached', snapshot_download(m))
import transformers, tokenizers
print('transformers', transformers.__version__, 'tokenizers', tokenizers.__version__)
"
