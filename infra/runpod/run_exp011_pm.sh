#!/usr/bin/env bash
# exp-011 addendum: PersonaManifold ORC pipeline and density-decoupling test on synthetic data.
# CPU work (GraphRicciCurvature needs Linux). Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && THREADS=28 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp011_pm.sh > /workspace/results/oddharak/exp-011-pm.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
OUT=/workspace/results/$PERSON/exp-011-synthetic-geometry/personamanifold
THREADS=${THREADS:-8}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
TERMINATE_DELAY=${TERMINATE_DELAY:-0}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run pytest -q tests/test_personamanifold_orc.py
  uv run python experiments/exp-011-synthetic-geometry/personamanifold_orc.py --out "$OUT" \
    --threads "$THREADS"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  sleep "$TERMINATE_DELAY"
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
