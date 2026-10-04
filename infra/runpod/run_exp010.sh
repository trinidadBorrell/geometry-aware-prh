#!/usr/bin/env bash
# exp-010: Levina-Bickel intrinsic dimension per layer vs n (see
# experiments/exp-010-intrinsic-dim). CPU pod. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp010.sh > /workspace/results/oddharak/exp-010.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
OUT=/workspace/results/$PERSON/exp-010-intrinsic-dim
MODELS=${MODELS:-bigscience/bloomz-560m}
THREADS=${THREADS:-$(nproc)}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python experiments/exp-010-intrinsic-dim/run.py --out "$OUT" --inventory > "$OUT/inventory.txt"
  uv run python experiments/exp-010-intrinsic-dim/run.py --out "$OUT" --models "$MODELS" --threads "$THREADS"
  uv run python experiments/exp-010-intrinsic-dim/plot.py --root "$OUT"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  # runpodctl config fails on this image unless its config file exists
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
