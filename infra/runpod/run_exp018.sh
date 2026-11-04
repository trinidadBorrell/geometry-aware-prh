#!/usr/bin/env bash
# exp-018 (emergence of LM x ViT alignment over pretraining, pilot) on one GPU pod:
#   0. checks exp-016's D2 captions, ViT features and geometry on the volume
#   1. features + bits per byte of every (series, checkpoint), one process each, plus the final
#      reference LM; weights download to the container disk and are removed after each checkpoint
#   2. ORC / density / TwoNN of the checkpoint d2s features (CPU workers)
#   3. checkpoint x ViT (k sweep at n = 10240) and checkpoint x reference pairs (GPU), summary
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=16 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp018.sh > /workspace/results/oddharak/exp-018.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
OUT=$RES/exp-018-alignment-emergence
V16=$RES/exp-016-positive-controls
X20=experiments/exp-020-lm-data-pilot
EXP=experiments/exp-018-alignment-emergence
SCRATCH_HF=${SCRATCH_HF:-/root/hf-scratch}
WORKERS=${WORKERS:-16}
TOKENS=${TOKENS:-16384}
KEEP_CACHES=${KEEP_CACHES:-2}
PERMUTATIONS=${PERMUTATIONS:-200}
EXTRACT_TIMEOUT=${EXTRACT_TIMEOUT:-30m}
SERIES=${SERIES:-"EleutherAI/pythia-160m EleutherAI/pythia-160m-seed1 EleutherAI/pythia-410m EleutherAI/pythia-410m-seed1"}
REVISIONS=${REVISIONS:-"step0 step16 step64 step256 step512 step1000 step2000 step4000 step8000 step16000 step32000 step64000 step100000 step143000"}
REFS=${REFS:-"EleutherAI/pythia-1.4b"}
VITS=${VITS:-"vit_giant_patch14_dinov2.lvd142m,vit_huge_patch14_clip_224.laion2b"}
CAPTIONS=${CAPTIONS:-orig,gemini}   # gemini only halves storage and time
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
  for v in ${VITS//,/ }; do
    for f in "features/${v}__cls__d2s.pt" "features/${v}__cls__d2l.pt" "geometry/${v}__cls__d2s.npz"; do
      [ -f "$V16/$f" ] || { echo "missing $V16/$f"; missing=1; }
    done
  done
  return $missing
}

extract_one() {  # repo [revision]
  local args=(--out "$OUT" --d2 "$V16/d2.parquet" --models "$1" --tokens "$TOKENS" --cache-dir "$SCRATCH_HF" --captions "$CAPTIONS")
  [ -n "${2:-}" ] && args+=(--revisions "$2")
  for attempt in 1 2 3; do
    if timeout "$EXTRACT_TIMEOUT" uv run python "$X20/extract_lm.py" "${args[@]}" >> "$RES/exp-018-extract.log" 2>&1; then
      rm -rf "${SCRATCH_HF:?}"/*
      return 0
    fi
    echo "$1 ${2:-}: attempt $attempt failed" >> "$RES/exp-018-extract.log"
  done
  rm -rf "${SCRATCH_HF:?}"/*
  return 1
}

extract_all() {
  for r in $REFS; do extract_one "$r" || { echo "extraction failed for $r"; return 1; }; done
  for s in $SERIES; do
    for rev in $REVISIONS; do
      extract_one "$s" "$rev" || { echo "extraction failed for $s@$rev"; return 1; }
    done
    echo "$s: features saved"
  done
}

job() {
  set -e
  check_inputs
  [ "${SKIP_EXTRACT:-0}" = "1" ] || extract_all
  du -sh "$OUT/features"
  cpu uv run python "$EXP/align.py" --roots "$OUT,$V16" --stages geometry --workers "$WORKERS" \
    --tmp /root/exp018_layers > "$RES/exp-018-geometry.log" 2>&1
  echo "geometry done"
  local refs=""
  for r in $REFS; do refs="${refs:+$refs,}${r//\//__}"; done
  uv run python "$EXP/align.py" --roots "$OUT,$V16" --stages pairs --vits "$VITS" --refs "$refs" --captions "$CAPTIONS" \
    --permutations "$PERMUTATIONS" --keep-caches "$KEEP_CACHES" > "$RES/exp-018-pairs.log" 2>&1
  echo "pairs done"
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
