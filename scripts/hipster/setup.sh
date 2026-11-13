#!/usr/bin/env bash
# One-off setup on the hipster login node (network I/O only, no compute), from ~/FlexSLM-medbench:
#  1. scratch directories and repo symlinks data/ hf_export/ results/ jobs/ -> scratch;
#  2. data streamed from DAS-6 (hipster reaches it over ssh): PMC tokens and the medmix SFT set;
#  3. the Qwen3 HF cache copied from DAS-6, and Intelligent-Internet/II-Medical-8B downloaded
#     (the candidate teacher for the 1.7B backbone, user decision 2026-11-13).
# NOTE (2026-11-13): this puts data and the HF cache in /scratch, which is wiped every few weeks. Run
# scripts/hipster/migrate_to_home.sbatch (capacity job) once no flexslm job is queued or running; if it
# has already run (~/flexslm_saves/cache/huggingface exists), do NOT re-run this script as is.
#   bash scripts/hipster/setup.sh
set -eo pipefail
S=/scratch/$USER/flexslm
D6=fs2.das6.science.uva.nl
D6S=/var/scratch/$USER/flexslm
mkdir -p "$S"/{hf_cache,store,data,hf_export,results,jobs}

cd "$HOME/FlexSLM-medbench"
for d in data hf_export results jobs; do
  if [[ ! -L $d ]]; then
    if [[ -d $d ]]; then cp -rn "$d"/. "$S/$d"/ && rm -rf "$d"; fi
    ln -s "$S/$d" "$d"
  fi
done

for d in pmc_qwen3 medmix_sft; do
  if [[ ! -f "$S/data/$d/.complete" ]]; then
    echo "[setup] streaming data/$d from DAS-6"
    mkdir -p "$S/data/$d"
    ssh -o BatchMode=yes "$D6" "tar -cf - -C $D6S/data/$d ." | tar -xf - -C "$S/data/$d"
    touch "$S/data/$d/.complete"
  fi
done

for m in models--Qwen--Qwen3-0.6B models--Qwen--Qwen3-1.7B; do
  if [[ ! -d "$S/hf_cache/hub/$m" ]]; then
    echo "[setup] copying $m from DAS-6"
    mkdir -p "$S/hf_cache/hub"
    ssh -o BatchMode=yes "$D6" "tar -cf - -C $D6S/hf_cache/hub $m" | tar -xf - -C "$S/hf_cache/hub"
  fi
done

export HF_HOME=$S/hf_cache HF_TOKEN=$(cat "$HOME/.hf_token" 2>/dev/null || true)
/home/$USER/miniconda3/envs/my_env/bin/python -c "
from huggingface_hub import snapshot_download
print('cached', snapshot_download('Intelligent-Internet/II-Medical-8B'))
"
du -sh "$S"/data/* "$S"/hf_cache
