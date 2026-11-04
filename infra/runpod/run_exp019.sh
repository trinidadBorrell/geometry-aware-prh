#!/usr/bin/env bash
# exp-019 (alignment under truncation) on one CPU pod: d2s pairs on the exp-016 features, then plots.
# The pod terminates itself after a copy window (COPY_WINDOW seconds, default 1800), whether the
# job succeeded or not; the log stays on the volume. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp019.sh > /workspace/results/oddharak/exp-019.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
OUT=$RES/exp-019-truncation/d2s
FEAT=$RES/exp-016-positive-controls/features
EXP=experiments/exp-019-truncation
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
COPY_WINDOW=${COPY_WINDOW:-1800}
PERMUTATIONS=${PERMUTATIONS:-200}
DRAWS=${DRAWS:-2}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_sha.txt"

job() {
  set -e
  uv run python "$EXP/truncate.py" d2s --features "$FEAT" --out "$OUT" \
    --permutations "$PERMUTATIONS" --draws "$DRAWS" --proj-draws 1 --layers 9
  uv run python "$EXP/plot.py" --root "$OUT"
  echo "d2s done"
}

( job ); status=$?
echo "job exit status: $status"
touch "$OUT/FINISHED_$status"
if [ "$AUTO_TERMINATE" = "1" ]; then
  echo "terminating in ${COPY_WINDOW}s"
  sleep "$COPY_WINDOW"
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
