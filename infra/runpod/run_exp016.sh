#!/usr/bin/env bash
# exp-016 (WP8 positive controls) on one GPU pod:
#   1. D2 = exp-006 WIT-1M subset + Gemini captions (prepare_d2.py)
#   2. features, one model per process (EXTRACT_TIMEOUT per attempt, 3 attempts), deleting each model's newly
#      downloaded weights from the shared HF cache as soon as its features are saved (storage is
#      limited; a model folder that existed before is kept)
#   3. per-layer ORC / density / TwoNN (CPU workers), then every control pair (GPU)
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=30 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp016.sh > /workspace/results/oddharak/exp-016.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
OUT=$RES/exp-016-positive-controls
EXP=experiments/exp-016-positive-controls
HUB=${HF_HOME:-/workspace/hf}/hub
WORKERS=${WORKERS:-24}
LLM_TOKENS=${LLM_TOKENS:-16384}  # tokens per LLM batch (4096 on a 32 GB GPU)
KEEP_CACHES=${KEEP_CACHES:-6}    # alignment caches held on the GPU (2 on a 32 GB GPU)
EXTRACT_TIMEOUT=${EXTRACT_TIMEOUT:-60m}  # per model and attempt (longer on GPUs under 32 GB)
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_sha.txt"
df -h /workspace | tail -1

uv pip install -q hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1 || export HF_HUB_ENABLE_HF_TRANSFER=0

cpu() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"; }

extract_all() {
  local models
  models=$(uv run python -c "
import importlib.util
s = importlib.util.spec_from_file_location('e', '$EXP/extract.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(' '.join([*m.TOWERS, *m.LLMS, *m.VITS, 'random']))")
  for m in $models; do
    local d existed=0 ok=1
    d=$(uv run python "$EXP/extract.py" --hub-dir "$m")
    [ -n "$d" ] && [ -d "$HUB/$d" ] && existed=1
    for attempt in 1 2 3; do
      if timeout "$EXTRACT_TIMEOUT" uv run python "$EXP/extract.py" --out "$OUT" --d2 "$OUT/d2.parquet" \
          --only "$m" --llm-tokens "$LLM_TOKENS" >> "$RES/exp-016-extract.log" 2>&1; then ok=0; break; fi
      echo "$m: attempt $attempt failed" >> "$RES/exp-016-extract.log"
      [ -n "$d" ] && rm -rf "$HUB/$d/blobs/"*.incomplete
    done
    [ "$ok" -eq 0 ] || { echo "extraction failed for $m"; return 1; }
    if [ -n "$d" ] && [ "$existed" -eq 0 ] && [ -d "$HUB/$d" ]; then
      rm -rf "${HUB:?}/$d"
      echo "$m: features saved, downloaded weights removed"
    else
      echo "$m: features saved"
    fi
  done
}

job() {
  set -e
  [ -f "$OUT/d2.parquet" ] || uv run python "$EXP/prepare_d2.py" \
    --src "$RES/exp-006-sample-size/wit_1m_first10240.parquet" --out "$OUT/d2.parquet"
  [ "${SKIP_EXTRACT:-0}" = "1" ] || extract_all
  du -sh "$OUT/features"
  cpu uv run python "$EXP/measure.py" --root "$OUT" --stages geometry --workers "$WORKERS" \
    > "$RES/exp-016-geometry.log" 2>&1
  echo "geometry done"
  uv run python "$EXP/measure.py" --root "$OUT" --stages pairs --keep-caches "$KEEP_CACHES" > "$RES/exp-016-pairs.log" 2>&1
  echo "pairs done"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
