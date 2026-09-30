#!/usr/bin/env bash
# exp-009: local CKA (Emily's cknna_local) vs the CKNNA definitions (see
# experiments/exp-009-local-cknna). GPU pod, >= 24 GB. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp009.sh > /workspace/results/oddharak/exp-009.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
OUT=/workspace/results/$PERSON/exp-009-local-cknna
PERMUTATIONS=${PERMUTATIONS:-200}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python experiments/exp-009-local-cknna/run.py --out "$OUT" --permutations "$PERMUTATIONS"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  # runpodctl config fails on this image unless its config file exists
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
