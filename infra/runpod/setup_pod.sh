#!/usr/bin/env bash
# One-time environment setup on a fresh pod (runs on the pod).
#   ssh -p <port> root@<ip> "bash /root/geometry-aware-prh/infra/runpod/setup_pod.sh"
# The venv lives on the container disk (fast, disposable); data and HF cache live on /workspace.
set -euo pipefail
. /etc/rp_environment
cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv sync --locked
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
