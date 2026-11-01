#!/usr/bin/env bash
# Phase B, second part (after run_exp015.sh, whose CPU jobs succeeded and whose extraction failed
# because the pod sets HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package):
#   1. exp-015 extraction, one model per process (45 min timeout, 3 attempts: a plain download
#      once stalled for good), deleting each newly downloaded model's weights from the shared HF
#      cache as soon as its features are cached (storage is limited; KEEP lists what was cached
#      before Phase B)
#   2. exp-014 per-layer ORC on the new poolings
#   3. exp-012 calibrated mutual kNN / CKA under the matched pairings:
#      token (LLM last x ViT CLS, plus T1 on last), mean (LLM avg x ViT avg, plus T2 on avg)
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=30 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp015b.sh > /workspace/results/oddharak/exp-015b.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
R15=$RES/exp-015-matched-pooling
HUB=${HF_HOME:-/workspace/hf}/hub
WORKERS=${WORKERS:-24}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
TERMINATE_DELAY=${TERMINATE_DELAY:-0}
KEEP="models--bigscience--bloomz-560m models--bigscience--bloomz-1b1 models--bigscience--bloomz-1b7
models--bigscience--bloomz-3b models--timm--vit_tiny_patch16_224.augreg_in21k
models--timm--vit_small_patch16_224.augreg_in21k models--timm--vit_base_patch16_224.mae
models--timm--vit_small_patch14_dinov2.lvd142m models--timm--vit_base_patch16_clip_224.laion2b"
mkdir -p "$R15"
git rev-parse HEAD > "$R15/git_sha_b.txt"

# fast multi-connection downloads (the pod image enables them but lacks the package)
uv pip install -q hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1 || export HF_HUB_ENABLE_HF_TRANSFER=0

cpu() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"; }

repo_dir() {  # HF cache folder of a model: org/name -> models--org--name, timm name -> models--timm--name
  case "$1" in */*) echo "models--${1//\//--}" ;; *) echo "models--timm--$1" ;; esac
}

extract_all() {
  local models
  models=$(uv run python -c "
import importlib.util, sys
s = importlib.util.spec_from_file_location('r', 'experiments/exp-012-id-mismatch/run.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(' '.join(m.LLMS + m.VITS))")
  for m in $models; do
    local ok=1
    for attempt in 1 2 3; do
      if timeout 45m uv run python experiments/exp-015-matched-pooling/extract.py --models "$m" \
          --pools last,avg >> "$RES/exp-015-extract.log" 2>&1; then ok=0; break; fi
      echo "$m: attempt $attempt failed" >> "$RES/exp-015-extract.log"
      rm -rf "$HUB/$(repo_dir "$m")/blobs/"*.incomplete
    done
    [ "$ok" -eq 0 ] || { echo "extraction failed for $m"; return 1; }
    local d
    d=$(repo_dir "$m")
    if [ -d "$HUB/$d" ] && ! grep -qw "$d" <<< "$KEEP"; then
      rm -rf "${HUB:?}/$d"
      echo "$m: features cached, weights removed"
    else
      echo "$m: features cached"
    fi
  done
}

job() {
  set -e
  [ "${SKIP_EXTRACT:-0}" = "1" ] || extract_all
  cpu uv run python experiments/exp-014-orc-paired/compute.py --out "$R15/orc" \
    --llm-pool last --vit-pool avg --workers "$WORKERS" > "$RES/exp-015-orc.log" 2>&1
  echo "orc done"
  uv run python experiments/exp-012-id-mismatch/run.py --out "$R15/align_token" --tracks T1,T3 \
    --llm-pool last --vit-pool cls > "$RES/exp-015-align-token.log" 2>&1
  echo "align token done"
  uv run python experiments/exp-012-id-mismatch/run.py --out "$R15/align_mean" --tracks T2,T3 \
    --llm-pool avg --vit-pool avg > "$RES/exp-015-align-mean.log" 2>&1
  echo "align mean done"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  sleep "$TERMINATE_DELAY"
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
