#!/usr/bin/env bash
# Null-calibrated PRH alignment (Groger et al. Algorithm 2) on the val modelset, then the
# raw-vs-calibrated figures from Aristotelian's own plotting (its Fig. 18/19 layout).
# GPU pod: the permutation null is dense n x n tensor work.
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_prh_calibration.sh > /workspace/results/oddharak/prh_calibration.log 2>&1 &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
export PERSON
OUT=/workspace/results/$PERSON/prh_val_calibrated
PERMUTATIONS=${PERMUTATIONS:-200}
DEVICE=${DEVICE:-cuda}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python -m geoprh.prh_calibration --out "$OUT" --modelset val \
    --permutations "$PERMUTATIONS" --device "$DEVICE"
  # Aristotelian's plotting reads prh_alignment*.npy straight out of that directory
  ( cd Aristotelian && uv run python -m scripts.plots.experiments \
      --sections prh_alignment --assets-dir "$OUT" )
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
