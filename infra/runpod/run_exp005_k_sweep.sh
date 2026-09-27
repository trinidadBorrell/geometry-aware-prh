#!/usr/bin/env bash
# exp-005: k sweep of mutual kNN and CKNNA (as published, Huh et al. Eqs. 16-18) with CKA as
# reference (see experiments/exp-005-cknna-large-k). CPU pod; size WORKERS by RAM (~3 GB each).
#   1. full val grid (all cached LLMs x ViTs), mKNN + CKNNA at every k, CKA once per pair
#   2. vision-vision: last-block CLS of every ViT against every other
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=8 nohup bash infra/runpod/run_exp005_k_sweep.sh > /workspace/results/oddharak/exp-005.log 2>&1 &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
export PERSON
OUT=/workspace/results/$PERSON/exp-005-cknna-large-k
WORKERS=${WORKERS:-10}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
SWEEP=experiments/exp-005-cknna-large-k/cknna_k_sweep.py

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python "$SWEEP" --out "$OUT/grid" --workers "$WORKERS"
  uv run python "$SWEEP" --out "$OUT/diagnostics" --llms diag --lvms diag --diagnostics \
    --workers "$WORKERS"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
