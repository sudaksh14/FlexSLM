#!/usr/bin/env bash
# Local (Windows, Git Bash) -> DAS-6 code sync of the working tree, including
# uncommitted and untracked-but-not-ignored files. Never copies paper/, data/,
# hf_export/, results/ or jobs/ (on DAS-6 those are symlinks to scratch).
# Converts shell/sbatch scripts to LF and sets exec bits on the cluster.
#   SSH_AUTH_SOCK=~/.ssh/agent.sock bash scripts/das6/sync.sh
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
HOST=skalra@fs2.das6.science.uva.nl
LIST=$(mktemp)
git ls-files -co --exclude-standard \
  | grep -v -E '^(paper|data|hf_export|results|jobs)/' \
  | while read -r f; do [[ -f "$f" ]] && echo "$f"; done > "$LIST"
tar -cf - -T "$LIST" | ssh -o BatchMode=yes "$HOST" \
  'mkdir -p ~/FlexSLM && cd ~/FlexSLM && tar -xf - && find . \( -name "*.sh" -o -name "*.sbatch" \) -exec sed -i "s/\r$//" {} + && chmod +x scripts/*.sh scripts/das6/*.sh scripts/das6/*.sbatch'
echo "synced $(wc -l < "$LIST") files to $HOST:~/FlexSLM"
rm -f "$LIST"
