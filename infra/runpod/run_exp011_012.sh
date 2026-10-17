#!/usr/bin/env bash
# exp-011 (WP0, synthetic manifolds, CPU) and exp-012 (WP1, per-layer-pair calibrated alignment
# vs ID mismatch, GPU) side by side on one GPU pod; see experiments/exp-011-*/ and exp-012-*/.
# Launch detached (THREADS = the pod's vCPUs; nproc reports the host's cores):
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && THREADS=16 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp011_012.sh > /workspace/results/oddharak/exp-011-012.log 2>&1 < /dev/null &"
# The pod removes itself only if both jobs succeed (after TERMINATE_DELAY seconds).
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
RES=/workspace/results/$PERSON
THREADS=${THREADS:-8}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
TERMINATE_DELAY=${TERMINATE_DELAY:-0}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

job012() {
  set -e
  local out=$RES/exp-012-id-mismatch
  mkdir -p "$out"
  git rev-parse HEAD > "$out/git_sha.txt"
  uv run python experiments/exp-012-id-mismatch/run.py --out "$out"
  uv run python experiments/exp-012-id-mismatch/analyze.py --root "$out" \
    --id-root "$RES/exp-010-intrinsic-dim"
}

job011() {
  set -e
  local out=$RES/exp-011-synthetic-geometry
  mkdir -p "$out"
  git rev-parse HEAD > "$out/git_sha.txt"
  uv run python experiments/exp-011-synthetic-geometry/run.py --out "$out" --threads "$THREADS"
}

( job012 ) > "$RES/exp-012.log" 2>&1 &
pid12=$!
( job011 ) > "$RES/exp-011.log" 2>&1 &
pid11=$!
wait $pid12; s12=$?
echo "exp-012 exit status: $s12"
wait $pid11; s11=$?
echo "exp-011 exit status: $s11"

if [ "$s11" -eq 0 ] && [ "$s12" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  sleep "$TERMINATE_DELAY"
  # runpodctl config fails on this image unless its config file exists
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit $(( s11 || s12 ))
