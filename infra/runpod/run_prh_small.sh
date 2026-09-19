#!/usr/bin/env bash
# Small-model PRH replication (platonic-rep + Aristotelian) on a GPU pod.
# Launch detached so it survives SSH disconnects:
#   ssh -p <port> root@<ip> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_prh_small.sh > /workspace/results/oddharak/prh_small.log 2>&1 &"
# Terminates its own pod at the end. With TERMINATE_ON_FAILURE=0 (default) a failed
# run leaves the pod up for inspection -- remember to terminate it by hand.
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
OUT=/workspace/results/$PERSON/prh_small
TERMINATE_ON_FAILURE=${TERMINATE_ON_FAILURE:-0}
mkdir -p "$OUT"

job() {
  set -e
  pushd platonic-rep >/dev/null
  uv run python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset small \
    --modality language --pool avg --batch_size 16 --output_dir "$OUT/platonic/features"
  uv run python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset small \
    --modality vision --pool cls --batch_size 64 --output_dir "$OUT/platonic/features"
  for metric in mutual_knn cka; do
    uv run python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset small \
      --modality_x language --pool_x avg --modality_y vision --pool_y cls --metric "$metric" \
      --input_dir "$OUT/platonic/features" --output_dir "$OUT/platonic/alignment"
  done
  popd >/dev/null
  uv run python experiments/prh_small/run_aristotelian.py --device cuda \
    --metrics mutual_knn,cka_lin --output-dir "$OUT/aristotelian"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] || [ "$TERMINATE_ON_FAILURE" = "1" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
