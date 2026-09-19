#!/usr/bin/env bash
# Set up (or update) the repo and environment on a pod, git-based. Run from your machine:
#   ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
# Clones/pulls the branch into /root/geometry-aware-prh (container disk) and syncs the locked
# uv environment. The pod only pulls; it never pushes. For uncommitted debugging changes use
# push_code.sh instead.
set -euo pipefail
. /etc/rp_environment
BRANCH=${BRANCH:-oddharak}
REPO_URL=${REPO_URL:-https://github.com/trinidadBorrell/geometry-aware-prh}
DEST=${DEST:-/root/geometry-aware-prh}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch --quiet origin "$BRANCH"
  git -C "$DEST" checkout --quiet "$BRANCH"
  git -C "$DEST" reset --quiet --hard "origin/$BRANCH"
else
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$DEST"
fi
cd "$DEST"
echo "code: $BRANCH @ $(git rev-parse --short HEAD)"

uv sync --locked
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
