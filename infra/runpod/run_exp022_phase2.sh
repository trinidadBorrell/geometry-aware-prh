#!/usr/bin/env bash
# exp-022 phase 2 on one GPU pod. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp022_phase2.sh > /workspace/results/oddharak/exp-022-phase2.log 2>&1 < /dev/null &"
# Always terminates its own pod GRACE seconds after the job ends (window to fetch results).
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
RES=/workspace/results/${PERSON:-oddharak}
OUT=$RES/exp-022-lexical-validation/phase2
PY_OUT=$OUT/pythia-orig
DOCCI=$OUT/docci
V16=$RES/exp-016-positive-controls
V18=$RES/exp-018-alignment-emergence
X20=experiments/exp-020-lm-data-pilot
SCRATCH_HF=${SCRATCH_HF:-/root/hf-scratch}
GRACE=${GRACE:-1800}
SERIES=${SERIES:-"EleutherAI/pythia-160m EleutherAI/pythia-160m-seed1 EleutherAI/pythia-410m EleutherAI/pythia-410m-seed1"}
REVISIONS=${REVISIONS:-"step0 step1000 step4000 step16000 step64000 step143000"}
VITS="vit_giant_patch14_dinov2.lvd142m,vit_huge_patch14_clip_224.laion2b"
mkdir -p "$OUT" "$PY_OUT" "$DOCCI" "$SCRATCH_HF"
git rev-parse HEAD > "$OUT/git_sha.txt"
cp experiments/exp-022-lexical-validation/phase2.py "$OUT/phase2.py.txt"
cp experiments/exp-020-lm-data-pilot/extract_lm.py "$OUT/extract_lm.py.txt"
export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=0
df -h /workspace | tail -1

pythia_orig() {
  set -e
  for s in $SERIES; do
    for rev in $REVISIONS; do
      uv run --no-sync python "$X20/extract_lm.py" --out "$PY_OUT" --d2 "$V16/d2.parquet" --models "$s" \
        --revisions "$rev" --captions orig --no-bpb --cache-dir "$SCRATCH_HF"
      rm -rf "${SCRATCH_HF:?}"/*
    done
  done
  uv run --no-sync python experiments/exp-018-alignment-emergence/align.py --roots "$PY_OUT,$V16" \
    --stages pairs --only d2l --vits "$VITS" --refs "" --captions orig --permutations 50 --keep-caches 2
  echo "pythia_orig done"
}

phase2() {
  timeout "${JOB_TIMEOUT:-5h}" uv run --no-sync python experiments/exp-022-lexical-validation/phase2.py \
    --out "$OUT" --features /root/p2_features --exp016 "$V16" --exp018 "$V18" --out-pythia "$PY_OUT" \
    --docci "$DOCCI" --cache-dir "$SCRATCH_HF"
}

export -f pythia_orig
export PY_OUT V16 X20 SCRATCH_HF SERIES REVISIONS VITS
timeout "${PYTHIA_TIMEOUT:-2h}" bash -c pythia_orig; a=$?
echo "pythia_orig exit status: $a"
phase2; b=$?
echo "job exit status: pythia_orig $a, phase2 $b"
sleep "$GRACE"
mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
