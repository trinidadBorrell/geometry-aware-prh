#!/usr/bin/env bash
# exp-031: word-shuffled WIT-1024 captions through PRH's 10 LLMs, then calibrated scoring, on one
# GPU pod. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp031.sh > /workspace/results/oddharak/exp-031.log 2>&1 < /dev/null &"
#
# Stages (STAGES, comma-separated): checks, extract, score, analyze, save.
#  - checks:  layer 0 unchanged by a token shuffle; re-extracted intact bloomz-560m reproduces the
#             cache; calibration reproduces one stored prh_val_calibrated pair. Any failure stops.
#  - extract: one process per (LLM, seed), with a timeout. Weights come from the volume's second
#             HF cache when present (offline), otherwise they are downloaded to container disk and
#             deleted after use. Features go to container disk (the volume is near quota).
#  - score:   prh_calibration on the shuffled features, rows straight to the volume (resumable).
#  - save:    copy the shuffled features into /workspace/activations (cache-first convention);
#             a quota failure here is logged, not fatal, because the scores are already saved.
# Terminates its own pod GRACE seconds after the job, whatever its exit status (set GRACE=-1 to
# keep the pod up).
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
PERSON=${PERSON:-oddharak}
export PERSON
RES=/workspace/results/$PERSON
OUT=$RES/exp-031-shuffled-capability
X=experiments/exp-031-shuffled-capability
SCRATCH=/root/exp031
TEXT_ROOT=$SCRATCH/activations
TEXTS=$SCRATCH/texts.json  # captions, read once online so offline model runs need no dataset
VOL_HUB=/workspace/.cache/huggingface/hub
STAGES=${STAGES:-checks,extract,score,analyze,save}
SEEDS=${SEEDS:-"0 1"}
GRACE=${GRACE:-600}
LLMS=${LLMS:-"bigscience/bloomz-560m bigscience/bloomz-1b1 bigscience/bloomz-1b7 bigscience/bloomz-3b bigscience/bloomz-7b1 openlm-research/open_llama_3b openlm-research/open_llama_7b openlm-research/open_llama_13b huggyllama/llama-7b huggyllama/llama-13b"}
mkdir -p "$OUT" "$SCRATCH" /root/hf-scratch
git rev-parse HEAD > "$OUT/git_sha.txt" 2>/dev/null || true
cp "$X/extract.py" "$OUT/extract.py.txt"; cp "$X/analyze.py" "$OUT/analyze.py.txt"
export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/root/hf-scratch  # datasets and any fallback downloads on container disk
df -h /workspace / | tail -2
cat /sys/fs/cgroup/memory.max 2>/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# Run one extractor command for a model: offline from the volume's HF cache if the model is
# there, otherwise download to container disk and delete the weights afterwards.
with_model() {
  local model=$1; shift
  local dir="models--${model//\//--}"
  if [ -d "$VOL_HUB/$dir/snapshots" ]; then
    HF_HUB_CACHE=$VOL_HUB HF_HUB_OFFLINE=1 timeout 50m "$@"
  else
    echo "[$model] not in $VOL_HUB, downloading to container disk"
    HF_HUB_CACHE=/root/hf-scratch/hub timeout 90m "$@"
    local st=$?
    rm -rf "/root/hf-scratch/hub/$dir"
    return $st
  fi
}

job() {
  set -e
  [ -s "$TEXTS" ] || uv run --no-sync python "$X/extract.py" texts --texts "$TEXTS"
  if has checks; then
    uv run --no-sync pytest -q -p no:warnings tests/test_exp031_shuffle.py
    with_model bigscience/bloomz-560m uv run --no-sync python "$X/extract.py" check-layer0 --texts "$TEXTS" \
      --model bigscience/bloomz-560m --out "$OUT"
    with_model huggyllama/llama-7b uv run --no-sync python "$X/extract.py" check-layer0 --texts "$TEXTS" \
      --model huggyllama/llama-7b --out "$OUT"
    with_model bigscience/bloomz-560m uv run --no-sync python "$X/extract.py" extract --texts "$TEXTS" \
      --model bigscience/bloomz-560m --transform reext --scratch "$SCRATCH" --out "$OUT"
    uv run --no-sync python "$X/extract.py" check-cache --model bigscience/bloomz-560m \
      --scratch "$SCRATCH" --root /workspace/activations --out "$OUT"
    uv run --no-sync python "$X/extract.py" check-score --root /workspace/activations \
      --stored-cal "$RES/prh_val_calibrated" --out "$OUT"
    echo "checks passed"
  fi
  if has extract; then
    for seed in $SEEDS; do
      for model in $LLMS; do
        done_flag="$SCRATCH/done/${model//\//__}_shufw$seed"
        if [ -e "$done_flag" ]; then echo "[$model shufw$seed] done, skipping"; continue; fi
        with_model "$model" uv run --no-sync python "$X/extract.py" extract --texts "$TEXTS" --model "$model" \
          --transform "shufw$seed" --scratch "$SCRATCH" --text-root "$TEXT_ROOT" --out "$OUT"
        rm -rf "$SCRATCH/shufw$seed/minhuh"  # upstream-format copy; the cache entry is kept
        mkdir -p "$SCRATCH/done" && touch "$done_flag"
      done
    done
  fi
  if has score; then
    for seed in $SEEDS; do
      uv run --no-sync python -m geoprh.prh_calibration --root /workspace/activations \
        --text-root "$TEXT_ROOT" --text-transform "shufw$seed" --out "$OUT/shufw$seed" \
        --metrics mutual_knn,cka_lin --k 10 --permutations 200
    done
  fi
  if has analyze; then
    uv run --no-sync python "$X/analyze.py" --out "$OUT" --stored "$RES/prh_val" \
      --stored-cal "$RES/prh_val_calibrated"
  fi
  if has save; then
    cp -rn "$TEXT_ROOT/." /workspace/activations/ \
      && echo "shuffled features saved to /workspace/activations" \
      || echo "WARNING: could not copy shuffled features to the volume (quota?); scores are saved"
  fi
}

( job ); status=$?
echo "job exit status: $status"
if [ "$GRACE" -ge 0 ]; then
  sleep "$GRACE"
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
