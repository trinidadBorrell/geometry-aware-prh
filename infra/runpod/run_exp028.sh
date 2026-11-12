#!/usr/bin/env bash
# exp-028 (spectral lexical floor) on a GPU or CPU pod. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp028.sh > /workspace/results/oddharak/exp-028.log 2>&1 < /dev/null &"
# Rows go straight to the volume. Always terminates its own pod GRACE seconds after the job ends.
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
RES=/workspace/results/${PERSON:-oddharak}
OUT=$RES/exp-028-spectral-lexical-floor
X=experiments/exp-028-spectral-lexical-floor
GRACE=${GRACE:-1200}
mkdir -p "$OUT" /root/hf-scratch
git rev-parse HEAD > "$OUT/git_sha.txt" 2>/dev/null || true
for f in spectral.py run.py; do cp "$X/$f" "$OUT/$f.txt"; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/hf-scratch UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=0

job() {
  set -e
  uv run --no-sync pytest -q tests/test_spectral_floor.py
  timeout 4h uv run --no-sync python "$X/run.py" --out "$OUT" \
    --exp016 "$RES/exp-016-positive-controls" --stored "$RES/exp-022-lexical-validation/rows.jsonl" \
    --stages "${STAGES:-anchors,kernels,score}"
}

( job ); status=$?
echo "job exit status: $status"
sleep "$GRACE"
mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
