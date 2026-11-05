#!/usr/bin/env bash
# exp-021 (word-count kernels and layer-0 grids, no network) on one CPU pod; CPU only, cached data.
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp021.sh > /workspace/results/oddharak/exp-021.log 2>&1 < /dev/null &"
# Always terminates its own pod GRACE seconds after the job ends (window to fetch results).
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
RES=/workspace/results/${PERSON:-oddharak}
OUT=$RES/exp-021-lexical-floor
GRACE=${GRACE:-1800}
mkdir -p "$OUT" /root/hf-scratch
git rev-parse HEAD > "$OUT/git_sha.txt"
cp experiments/exp-021-lexical-floor/run.py "$OUT/run.py.txt"
export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} HF_HUB_ENABLE_HF_TRANSFER=0
timeout "${JOB_TIMEOUT:-4h}" uv run python experiments/exp-021-lexical-floor/run.py --out "$OUT" \
  --exp016 "$RES/exp-016-positive-controls" --exp018 "$RES/exp-018-alignment-emergence" \
  --cache-dir /root/hf-scratch
echo "job exit status: $?"
sleep "$GRACE"
mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
