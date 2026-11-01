#!/usr/bin/env bash
# Phase B on one GPU pod (see experiments/exp-015-matched-pooling, exp-014-orc-paired):
#   GPU:  exp-015 extraction of LLM last-token and ViT patch-mean features (wit_1024, 27 models)
#   CPU, in parallel: exp-014 per-sample check on all T1/T2 pairs, exp-014 k-sweep
#   then: exp-014 per-layer ORC on the new poolings
# Needs results/oddharak/exp-014-orc-paired/pairs.jsonl on the volume (copied from local).
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=30 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp015.sh > /workspace/results/oddharak/exp-015.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
R14=$RES/exp-014-orc-paired
R15=$RES/exp-015-matched-pooling
WORKERS=${WORKERS:-24}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
TERMINATE_DELAY=${TERMINATE_DELAY:-0}
mkdir -p "$R15"
git rev-parse HEAD > "$R15/git_sha.txt"

cpu() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"; }

( set -e; cpu uv run python experiments/exp-014-orc-paired/per_sample.py --root "$R14" \
    --cache /workspace/activations ) > "$RES/exp-014-per-sample.log" 2>&1 &
p_ps=$!
( set -e; cpu uv run python experiments/exp-014-orc-paired/k_sweep.py --out "$R14/k_sweep" \
    --workers $((WORKERS - 6)) ) > "$RES/exp-014-ksweep.log" 2>&1 &
p_ks=$!
( set -e; uv run python experiments/exp-015-matched-pooling/extract.py --pools last,avg ) \
    > "$RES/exp-015-extract.log" 2>&1
s_ex=$?
echo "extraction exit status: $s_ex"
wait $p_ks; s_ks=$?
echo "k-sweep exit status: $s_ks"
s_orc=1
if [ "$s_ex" -eq 0 ]; then
  ( set -e; cpu uv run python experiments/exp-014-orc-paired/compute.py --out "$R15/orc" \
      --llm-pool last --vit-pool avg --workers "$WORKERS" ) > "$RES/exp-015-orc.log" 2>&1
  s_orc=$?
fi
echo "orc exit status: $s_orc"
wait $p_ps; s_ps=$?
echo "per-sample exit status: $s_ps"
status=$(( s_ex || s_ks || s_orc || s_ps ))
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  sleep "$TERMINATE_DELAY"
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
