#!/usr/bin/env bash
# exp-027 (I-JEPA vs MAE against LLMs) on one GPU pod. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp027.sh > /workspace/results/oddharak/exp-027.log 2>&1 < /dev/null &"
# Rows go straight to the volume. Always terminates its own pod GRACE seconds after the job ends.
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
RES=/workspace/results/${PERSON:-oddharak}
OUT=$RES/exp-027-ijepa-vs-llm
V16=$RES/exp-016-positive-controls
X=experiments/exp-027-ijepa-vs-llm
GRACE=${GRACE:-1200}
mkdir -p "$OUT" /root/hf-scratch
git rev-parse HEAD > "$OUT/git_sha.txt" 2>/dev/null || true
for f in extract.py score_d2.py wit.py; do cp "$X/$f" "$OUT/$f.txt"; done
# the volume is near quota: model weights and datasets go to the container disk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/hf-scratch UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=0
df -h /workspace | tail -1

job() {
  set -e
  # ViT-g with all 40 block outputs kept does not fit batch 64 on a 24 GB GPU
  timeout 2h uv run --no-sync python "$X/extract.py" --out "$OUT" --d2 "$V16/d2.parquet" \
    --batch "${BATCH:-64}"
  timeout 2h uv run --no-sync python "$X/score_d2.py" --out "$OUT" --exp016 "$V16" \
    --grids "$RES/exp-022-lexical-validation/grids_full.json" --group "${GROUP:-2}"
  timeout 3h uv run --no-sync python "$X/wit.py" --out "$OUT" --stored-cal "$RES/prh_val_calibrated"
}

( job ); status=$?
echo "job exit status: $status"
df -h /workspace | tail -1
sleep "$GRACE"
mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
