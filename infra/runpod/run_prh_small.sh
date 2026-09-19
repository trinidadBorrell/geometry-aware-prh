#!/usr/bin/env bash
# Small-model PRH replication (platonic-rep + Aristotelian) on a GPU pod, cache first.
#   1. pull cached activations from /workspace/activations into upstream feature files
#   2. extract only what is missing (upstream scripts skip files that exist)
#   3. push any new activations back to the shared cache
#   4. alignment: platonic-rep raw + Aristotelian null-calibrated
# Launch detached so it survives SSH disconnects:
#   ssh -p <port> root@<ip> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_prh_small.sh > /workspace/results/oddharak/prh_small.log 2>&1 &"
# With AUTO_TERMINATE=1 the pod removes itself after a successful run; a failed run always
# leaves the pod up for inspection -- remember to terminate it by hand.
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
# The Runpod image enables hf_transfer, which we do not install (hf-xet handles fast downloads).
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
export PERSON
OUT=/workspace/results/$PERSON/prh_small
PLAT=$OUT/platonic/features/minhuh/prh/wit_1024
ARIST=$OUT/aristotelian/features/minhuh/prh/wit_1024
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
cache() { uv run python -m geoprh.activation_cache "$@" --modelset small; }

job() {
  set -e
  cache pull --layout platonic --dir "$PLAT"
  cache pull --layout aristotelian --dir "$ARIST"

  pushd platonic-rep >/dev/null
  uv run python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset small \
    --modality language --pool avg --batch_size 16 --output_dir "$OUT/platonic/features"
  uv run python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset small \
    --modality vision --pool cls --batch_size 64 --output_dir "$OUT/platonic/features"
  popd >/dev/null
  cache push --layout platonic --dir "$PLAT"
  # platonic-rep files now cover every model; give Aristotelian the same activations
  cache pull --layout aristotelian --dir "$ARIST"

  pushd platonic-rep >/dev/null
  for metric in mutual_knn cka; do
    uv run python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset small \
      --modality_x language --pool_x avg --modality_y vision --pool_y cls --metric "$metric" \
      --input_dir "$OUT/platonic/features" --output_dir "$OUT/platonic/alignment"
  done
  popd >/dev/null
  uv run python experiments/prh_small/run_aristotelian.py --device cuda \
    --metrics mutual_knn,cka_lin --output-dir "$OUT/aristotelian"
  uv run python experiments/prh_small/compare.py --root "$OUT"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
