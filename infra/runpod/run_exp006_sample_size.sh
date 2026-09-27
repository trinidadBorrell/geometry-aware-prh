#!/usr/bin/env bash
# exp-006: sample size and sample-set variability (see experiments/exp-006-sample-size).
#   1. extract 10240 WIT-1M samples for 3 LLMs x 5 ViTs into the shared activation cache
#      (cached models are skipped)
#   2. Groger-calibrated mKNN and CKA on 5 disjoint 1024-sample sets and n = 2048, 4096, 10240
# GPU pod (forward passes, then GPU matmuls for the nulls). Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp006_sample_size.sh > /workspace/results/oddharak/exp-006.log 2>&1 &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
export PERSON
OUT=/workspace/results/$PERSON/exp-006-sample-size
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
EXP=experiments/exp-006-sample-size

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python "$EXP/extract.py" --n 10240 --data "$OUT/wit_1m_first10240.parquet"
  uv run python "$EXP/calibrate.py" --out "$OUT"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
