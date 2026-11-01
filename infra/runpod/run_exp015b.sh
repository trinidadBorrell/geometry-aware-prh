#!/usr/bin/env bash
# Phase B, second part (after run_exp015.sh, whose CPU jobs succeeded and whose extraction failed
# because the pod sets HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package):
#   1. exp-015 extraction (downloads the uncached models into the shared HF cache)
#   2. exp-014 per-layer ORC on the new poolings
#   3. exp-012 calibrated mutual kNN / CKA under the matched pairings:
#      token (LLM last x ViT CLS, plus T1 on last), mean (LLM avg x ViT avg, plus T2 on avg)
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=30 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp015b.sh > /workspace/results/oddharak/exp-015b.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
R15=$RES/exp-015-matched-pooling
WORKERS=${WORKERS:-24}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
TERMINATE_DELAY=${TERMINATE_DELAY:-0}
mkdir -p "$R15"
git rev-parse HEAD > "$R15/git_sha_b.txt"

cpu() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"; }

job() {
  set -e
  uv run python experiments/exp-015-matched-pooling/extract.py --pools last,avg \
    > "$RES/exp-015-extract.log" 2>&1
  echo "extraction done"
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
