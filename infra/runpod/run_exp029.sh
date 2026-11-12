#!/usr/bin/env bash
# exp-029 (word vectors from captions vs general text) on a CPU pod. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp029.sh > /workspace/results/oddharak/exp-029.log 2>&1 < /dev/null &"
# Every stage writes to the volume and is resumable. Terminates its own pod GRACE seconds after the job.
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
RES=/workspace/results/${PERSON:-oddharak}
OUT=$RES/exp-029-ood-word-vectors
X=experiments/exp-029-ood-word-vectors
E28=$RES/exp-028-spectral-lexical-floor
GRACE=${GRACE:-600}
mkdir -p "$OUT" /root/hf-scratch
git rev-parse HEAD > "$OUT/git_sha.txt" 2>/dev/null || true
for f in run.py words.py; do cp "$X/$f" "$OUT/$f.txt"; done
export HF_HOME=/root/hf-scratch UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=0
df -h /workspace | tail -1

job() {
  set -e
  uv run --no-sync pytest -q tests/test_ood_words.py tests/test_spectral_floor.py
  timeout 4h uv run --no-sync python "$X/run.py" --out "$OUT" --exp016 "$RES/exp-016-positive-controls" \
    --anchors "$E28/anchors.parquet" --exp028-rows "$E28/rows.jsonl" \
    --stored "$RES/exp-022-lexical-validation/rows.jsonl" --stages "${STAGES:-corpora,vectors,kernels,score,check}"
}

( job ); status=$?
echo "job exit status: $status"
sleep "$GRACE"
mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
