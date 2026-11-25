#!/usr/bin/env bash
#SBATCH --job-name=flexslm_to_das6
#SBATCH --partition=capacity
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
# Expected (pessimistic) 45 min: ~10 GB streamed to DAS-6, one connection at a time (DAS-6 throttles bursts).
#SBATCH --time=01:30:00
#SBATCH --output=jobs/%x-%j.out
# MOVES HF exports from hf_export/ (hipster /scratch) to DAS-6 /var/scratch/$USER/flexslm/hf_export/<same name>.
# Per export: tar stream to DAS-6, SHA-256 of the stream on both ends, then per-file sizes compared. Only when
# both match is the hipster copy deleted. Refuses to overwrite an existing export on DAS-6.
#   sbatch scripts/hipster/move_exports_das6.sbatch <export dir relative to hf_export/> ...
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/FlexSLM-medbench}"
D6=fs2.das6.science.uva.nl
DST=/var/scratch/$USER/flexslm/hf_export
SSH=(ssh -o BatchMode=yes -o LogLevel=QUIET "$D6")
for d in "$@"; do
  src="hf_export/$d"
  [[ -d "$src" ]] || { echo "[move] missing $src"; exit 2; }
  if "${SSH[@]}" "test -e $DST/$d"; then echo "[move] $DST/$d already exists on DAS-6, not touching it"; exit 3; fi
  echo "[move] $d ($(du -sh "$src" | cut -f1))"
  local_sum=$(mktemp)
  tar -cf - -C "$src" . | tee >(sha256sum | cut -d' ' -f1 > "$local_sum") \
    | "${SSH[@]}" "mkdir -p $DST/$d.partial && tee >(sha256sum | cut -d' ' -f1 > $DST/$d.partial/.stream_sha256) | tar -xf - -C $DST/$d.partial"
  sleep 2   # let the process substitutions finish writing
  remote_sha=$("${SSH[@]}" "cat $DST/$d.partial/.stream_sha256")
  local_sha=$(cat "$local_sum"); rm -f "$local_sum"
  [[ -n "$local_sha" && "$local_sha" == "$remote_sha" ]] || { echo "[move] STREAM HASH MISMATCH $d: $local_sha vs $remote_sha"; exit 4; }
  local_list=$(cd "$src" && find . -type f -printf '%P %s\n' | sort)
  remote_list=$("${SSH[@]}" "cd $DST/$d.partial && find . -type f ! -name .stream_sha256 -printf '%P %s\n' | sort")
  [[ "$local_list" == "$remote_list" ]] || { echo "[move] FILE LIST/SIZE MISMATCH $d"; diff <(echo "$local_list") <(echo "$remote_list") || true; exit 5; }
  "${SSH[@]}" "mv $DST/$d.partial $DST/$d && touch $DST/$d/.complete"
  echo "[move] $d verified on DAS-6 ($(echo "$local_list" | wc -l) files, sha256 $local_sha); deleting hipster copy"
  rm -rf "$src"
done
"${SSH[@]}" "ls -d $DST/flexllama__sft.paper.qwen3_0p6b_nested.no_stage1*"
echo "[move] done"
