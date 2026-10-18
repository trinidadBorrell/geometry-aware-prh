#!/usr/bin/env bash
# exp-013: moment curvature estimator on synthetic manifolds, then on the 8 wit1m10k models.
# CPU work; any pod with many vCPUs and ~60 GB RAM. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp013.sh > /workspace/results/oddharak/exp-013.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
OUT=/workspace/results/$PERSON/exp-013-curvature-moments
SYN_WORKERS=${SYN_WORKERS:-12}
MODEL_WORKERS=${MODEL_WORKERS:-8}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
TERMINATE_DELAY=${TERMINATE_DELAY:-0}

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python experiments/exp-013-curvature-moments/run.py --mode synthetic --out "$OUT" \
    --workers "$SYN_WORKERS"
  uv run python experiments/exp-013-curvature-moments/run.py --mode models --out "$OUT" \
    --workers "$MODEL_WORKERS"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  sleep "$TERMINATE_DELAY"
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
