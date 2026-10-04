#!/usr/bin/env bash
# exp-010: Levina-Bickel intrinsic dimension per layer (see experiments/exp-010-intrinsic-dim).
# CPU pod. STAGE=1: MODELS vs n; STAGE=2: every model at n=1024 (PRH wit_1024), plus the
# other wit1m10k models at n=1000/10000 to check the model ordering; STAGE=all: both, with all 8
# wit1m10k models at every n. ESTIMATOR=lb (Levina-Bickel) or twonn (outputs under twonn/).
# Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && STAGE=2 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp010.sh > /workspace/results/oddharak/exp-010.log 2>&1 < /dev/null &"
# TERMINATE_DELAY seconds pass between the job's end and the pod removing itself, so the
# results can be copied off first.
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
ESTIMATOR=${ESTIMATOR:-lb}
OUT=/workspace/results/$PERSON/exp-010-intrinsic-dim
LB_OUT=$OUT
[ "$ESTIMATOR" = "lb" ] || OUT=$OUT/$ESTIMATOR
STAGE=${STAGE:-1}
MODELS=${MODELS:-bigscience/bloomz-560m}
CHECK_MODELS=${CHECK_MODELS:-bigscience/bloomz-1b1,bigscience/bloomz-1b7,vit_tiny_patch16_224.augreg_in21k,vit_small_patch16_224.augreg_in21k,vit_base_patch16_224.mae,vit_small_patch14_dinov2.lvd142m,vit_base_patch16_clip_224.laion2b}
THREADS=${THREADS:-$(nproc)}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
TERMINATE_DELAY=${TERMINATE_DELAY:-0}
RUN="uv run python experiments/exp-010-intrinsic-dim"
EST="--estimator $ESTIMATOR"

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  if [ "$STAGE" = "1" ]; then
    $RUN/run.py --out "$OUT" --inventory > "$OUT/inventory.txt"
    $RUN/run.py --out "$OUT" --models "$MODELS" --threads "$THREADS" $EST
    $RUN/plot.py --root "$OUT"
    return
  fi
  if [ "$STAGE" = "all" ]; then
    $RUN/run.py --out "$OUT" --models "$MODELS,$CHECK_MODELS" --threads "$THREADS" $EST
  else
    $RUN/run.py --out "$OUT" --models "$CHECK_MODELS" --sizes 1000,10000 --threads "$THREADS" $EST
  fi
  mkdir -p "$OUT/prh1024"
  $RUN/run.py --out "$OUT/prh1024" --data prh1024 --models all --sizes 1024 --threads "$THREADS" $EST
  $RUN/plot.py --root "$OUT"
  OTHER=""
  [ "$OUT" != "$LB_OUT" ] && [ -f "$LB_OUT/prh1024/intrinsic_dim.jsonl" ] && OTHER="--other $LB_OUT/prh1024"
  $RUN/compare.py --root "$OUT/prh1024" --check "$OUT" $OTHER
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  sleep "$TERMINATE_DELAY"
  # runpodctl config fails on this image unless its config file exists
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
