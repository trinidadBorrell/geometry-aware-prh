#!/usr/bin/env bash
# PRH (Huh et al.) Fig. 3, 10, 12, 13, 14 on the full "val" modelset, CPU only.
#   1. copy any val-set feature files from SOURCES into the shared activation cache
#      (read-only on the sources; the cache never overwrites existing entries)
#   2. cross-modal alignment (all metrics, max over layer pairs), vision-vision alignment
#   3. figures
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_prh_val.sh > /workspace/results/oddharak/prh_val.log 2>&1 &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
export PERSON
OUT=/workspace/results/$PERSON/prh_val
WORKERS=${WORKERS:-$(nproc)}
# existing platonic-rep feature dirs to import into the cache (space separated)
SOURCES=${SOURCES:-"/workspace/hf/prh_llms/wit_1024 /workspace/hf/prh_vlms/prh/wit_1024"}
EXTRACTOR=${EXTRACTOR:-"platonic-rep extract_features.py (run by emily, 2026-09-17/18)"}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  # shellcheck disable=SC2086
  uv run python -m geoprh.activation_cache push --modelset val --layout platonic \
    --dir $SOURCES --extractor "$EXTRACTOR"
  uv run python -m geoprh.activation_cache ls --modelset val
  uv run python -m geoprh.prh_alignment vision --modelset val --out "$OUT"
  uv run python -m geoprh.prh_alignment cross --modelset val --out "$OUT" --workers "$WORKERS"
  uv run python experiments/prh_val/plot_figures.py --root "$OUT"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
