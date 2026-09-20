#!/usr/bin/env bash
# exp-004: neighbourhood size k for mutual kNN and CKNNA (see experiments/exp-004-knn-k-sweep).
# Calibrates both metrics at k=200 on the reduced grid, then draws Aristotelian's raw-vs-calibrated
# figures. GPU pod: CKNNA's null loop is a batched n x n x n matmul.
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp004_k_sweep.sh > /workspace/results/oddharak/exp-004.log 2>&1 &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
export PERSON
EXP=exp-004-knn-k-sweep
OUT=/workspace/results/$PERSON/$EXP
K=${K:-200}
METRICS=${METRICS:-mutual_knn,cknna}
MODELS=${MODELS:-reduced}
PERMUTATIONS=${PERMUTATIONS:-200}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python -m geoprh.prh_calibration --out "$OUT" --modelset val --models "$MODELS" \
    --metrics "$METRICS" --k "$K" --permutations "$PERMUTATIONS" --device cuda
  ( cd Aristotelian && uv run python -m scripts.plots.experiments \
      --sections prh_alignment --assets-dir "$OUT" )
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
