#!/usr/bin/env bash
# exp-007 (CKNNA variants k sweep) and exp-008 (their permutation-null calibration), see
# experiments/exp-007-cknna-variants and experiments/exp-008-cknna-null. GPU pod, >= 24 GB.
#   1. exp-007: full val grid, all variants + mKNN at every k, CKA once per pair
#   2. exp-007: vision-vision, last-block CLS of every ViT against every other
#   3. exp-008: the same grid, 200 permutations per pair, every k
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp007_exp008.sh > /workspace/results/oddharak/exp-007-008.log 2>&1 &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
export PERSON
OUT7=/workspace/results/$PERSON/exp-007-cknna-variants
OUT8=/workspace/results/$PERSON/exp-008-cknna-null
PERMUTATIONS=${PERMUTATIONS:-200}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}

job() {
  set -e
  mkdir -p "$OUT7" "$OUT8"
  git rev-parse HEAD | tee "$OUT7/git_sha.txt" > "$OUT8/git_sha.txt"
  uv run python experiments/exp-007-cknna-variants/sweep.py --out "$OUT7/grid"
  uv run python experiments/exp-007-cknna-variants/sweep.py --out "$OUT7/vision_vision" --vision-vision
  uv run python experiments/exp-008-cknna-null/calibrate.py --out "$OUT8" --permutations "$PERMUTATIONS"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  # runpodctl config fails on this image unless its config file exists (2026-09-29 run)
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
