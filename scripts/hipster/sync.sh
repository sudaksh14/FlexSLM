#!/usr/bin/env bash
# Local (Windows, Git Bash) -> hipster code sync of the working tree into ~/FlexSLM-medbench
# (same file list as scripts/das6/sync.sh; never paper/, data/, hf_export/, results/, jobs/).
#   SSH_AUTH_SOCK=~/.ssh/agent.sock bash scripts/hipster/sync.sh
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
HOST=hipster
LIST=$(mktemp)
git ls-files -co --exclude-standard \
  | grep -v -E '^(paper|data|hf_export|results|jobs)/' \
  | while read -r f; do [[ -f "$f" ]] && echo "$f"; done > "$LIST"
tar -cf - -T "$LIST" | ssh -o BatchMode=yes -o LogLevel=QUIET "$HOST" \
  'mkdir -p ~/FlexSLM-medbench && cd ~/FlexSLM-medbench && tar -xf - && find . \( -name "*.sh" -o -name "*.sbatch" \) -exec sed -i "s/\r$//" {} + && chmod +x scripts/*.sh scripts/das6/*.sh scripts/hipster/*.sh scripts/*/*.sbatch'
echo "synced $(wc -l < "$LIST") files to $HOST:~/FlexSLM-medbench"
rm -f "$LIST"
