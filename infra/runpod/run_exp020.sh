#!/usr/bin/env bash
# exp-020 (LM training data vs alignment with vision, pilot) on one GPU pod:
#   0. checks that exp-016's D2 captions, ViT features (d2s, d2l) and ViT geometry are on the volume
#   1. LM features + bits per byte, one model per process (EXTRACT_TIMEOUT per attempt, 3 attempts);
#      weights download to the container disk ($SCRATCH_HF) and are removed after each model, so the
#      shared HF cache and the volume quota are untouched
#   2. ORC / density / TwoNN of the LM d2s features (CPU workers)
#   3. LM x ViT and base x continuation pairs (GPU), matched contrasts, summary
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=16 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp020.sh > /workspace/results/oddharak/exp-020.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
OUT=$RES/exp-020-lm-data-pilot
V16=$RES/exp-016-positive-controls
EXP=experiments/exp-020-lm-data-pilot
SCRATCH_HF=${SCRATCH_HF:-/root/hf-scratch}
WORKERS=${WORKERS:-16}
TOKENS=${TOKENS:-16384}           # tokens per LM batch
KEEP_CACHES=${KEEP_CACHES:-2}     # alignment caches on the GPU (n = 10240: ~8 GB each)
EXTRACT_TIMEOUT=${EXTRACT_TIMEOUT:-60m}
MODELS=${MODELS:-"Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-Coder-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-Math-1.5B Qwen/Qwen2-0.5B-Instruct llava-hf/llava-onevision-qwen2-0.5b-ov-hf HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolVLM-256M-Instruct"}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
mkdir -p "$OUT" "$SCRATCH_HF"
git rev-parse HEAD > "$OUT/git_sha.txt"
df -h /workspace | tail -1
export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy

uv pip install -q hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1 || export HF_HUB_ENABLE_HF_TRANSFER=0

cpu() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"; }

check_inputs() {
  local missing=0
  [ -f "$V16/d2.parquet" ] || { echo "missing $V16/d2.parquet"; missing=1; }
  for v in vit_large_patch16_224.augreg_in21k vit_huge_patch14_224.mae vit_giant_patch14_dinov2.lvd142m vit_huge_patch14_clip_224.laion2b; do
    for f in "features/${v}__cls__d2s.pt" "features/${v}__cls__d2l.pt" "geometry/${v}__cls__d2s.npz"; do
      [ -f "$V16/$f" ] || { echo "missing $V16/$f"; missing=1; }
    done
  done
  return $missing
}

extract_all() {
  for m in $MODELS; do
    local ok=1
    for attempt in 1 2 3; do
      if timeout "$EXTRACT_TIMEOUT" uv run python "$EXP/extract_lm.py" --out "$OUT" --d2 "$V16/d2.parquet" \
          --models "$m" --tokens "$TOKENS" --cache-dir "$SCRATCH_HF" >> "$RES/exp-020-extract.log" 2>&1; then ok=0; break; fi
      echo "$m: attempt $attempt failed" >> "$RES/exp-020-extract.log"
    done
    rm -rf "${SCRATCH_HF:?}"/*
    [ "$ok" -eq 0 ] || { echo "extraction failed for $m"; return 1; }
    echo "$m: features saved"
  done
}

job() {
  set -e
  check_inputs
  [ "${SKIP_EXTRACT:-0}" = "1" ] || extract_all
  du -sh "$OUT/features"
  cpu uv run python "$EXP/align.py" --roots "$OUT,$V16" --stages geometry --workers "$WORKERS" \
    --tmp /root/exp020_layers > "$RES/exp-020-geometry.log" 2>&1
  echo "geometry done"
  uv run python "$EXP/align.py" --roots "$OUT,$V16" --stages pairs,contrasts \
    --keep-caches "$KEEP_CACHES" > "$RES/exp-020-pairs.log" 2>&1
  echo "pairs and contrasts done"
  uv run python "$EXP/analyze.py" --out "$OUT" > /dev/null
  echo "summary written"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
