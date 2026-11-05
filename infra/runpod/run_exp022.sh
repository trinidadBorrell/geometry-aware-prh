#!/usr/bin/env bash
# exp-022 phase 1 (is the lexical floor real?) on one CPU pod: cached data, no forward passes.
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp022.sh > /workspace/results/oddharak/exp-022.log 2>&1 < /dev/null &"
# Always terminates its own pod GRACE seconds after the job ends (window to fetch results).
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
RES=/workspace/results/${PERSON:-oddharak}
OUT=$RES/exp-022-lexical-validation
GRACE=${GRACE:-1800}
SCRATCH_HF=${SCRATCH_HF:-/root/hf-scratch}
mkdir -p "$OUT" "$SCRATCH_HF"
git rev-parse HEAD > "$OUT/git_sha.txt"
export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
uv pip install -q langdetect || echo "langdetect not installed: language check will be skipped"
timeout "${JOB_TIMEOUT:-3h}" uv run --no-sync python experiments/exp-022-lexical-validation/run.py \
  --out "$OUT" --exp016 "$RES/exp-016-positive-controls" --exp018 "$RES/exp-018-alignment-emergence" \
  --cache-dir "$SCRATCH_HF"
echo "job exit status: $?"
sleep "$GRACE"
mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
