#!/usr/bin/env bash
# exp-017 (map complexity) on one GPU pod: real pairs on the exp-016 n = 10240 features, then the
# synthetic power check. Resumable (skips finished pair / direction / fold rows).
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp017.sh > /workspace/results/oddharak/exp-017.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
OUT=$RES/exp-017-map-complexity
FEAT=$RES/exp-016-positive-controls/features
EXP=experiments/exp-017-map-complexity
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_sha.txt"

job() {
  set -e
  uv run python "$EXP/maps.py" real --features "$FEAT" --out "$OUT"
  echo "real done"
  uv run python "$EXP/maps.py" synthetic --out "$OUT"
  echo "synthetic done"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
