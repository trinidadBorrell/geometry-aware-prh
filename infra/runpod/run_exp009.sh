#!/usr/bin/env bash
# exp-009: local CKA (Emily's cknna_local) vs the CKNNA definitions (see
# experiments/exp-009-local-cknna). GPU pod, >= 24 GB. Launch detached:
#   ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp009.sh > /workspace/results/oddharak/exp-009.log 2>&1 < /dev/null &"
set -uo pipefail
. /etc/rp_environment
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

PERSON=${PERSON:-oddharak}
OUT=/workspace/results/$PERSON/exp-009-local-cknna
PERMUTATIONS=${PERMUTATIONS:-200}
AUTO_TERMINATE=${AUTO_TERMINATE:-0}
# default: 5 LLMs (0.56B-13B, three families) x one large ViT per vision family = 25 pairs
LLMS=${LLMS:-bigscience/bloomz-560m,bigscience/bloomz-7b1,openlm-research/open_llama_3b,openlm-research/open_llama_13b,huggyllama/llama-13b}
LVMS=${LVMS:-vit_large_patch16_224.augreg_in21k,vit_large_patch16_224.mae,vit_large_patch14_dinov2.lvd142m,vit_large_patch14_clip_224.laion2b,vit_large_patch14_clip_224.laion2b_ft_in12k}

job() {
  set -e
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/git_sha.txt"
  uv run python experiments/exp-009-local-cknna/run.py --out "$OUT" --permutations "$PERMUTATIONS"     --llms "$LLMS" --lvms "$LVMS"
}

( job ); status=$?
echo "job exit status: $status"
if [ "$status" -eq 0 ] && [ "$AUTO_TERMINATE" = "1" ]; then
  # runpodctl config fails on this image unless its config file exists
  mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID"
fi
exit "$status"
