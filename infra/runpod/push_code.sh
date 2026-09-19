#!/usr/bin/env bash
# Copy the working tree (tracked + untracked, minus .gitignore'd files) to a pod.
# Run locally from Git Bash:  infra/runpod/push_code.sh <ip> <port>
# Uses the pod's *direct* SSH address (not ssh.runpod.io, which rejects commands).
set -euo pipefail
IP=${1:?usage: push_code.sh <ip> <port>}
PORT=${2:?usage: push_code.sh <ip> <port>}
REMOTE_DIR=${REMOTE_DIR:-/root/geometry-aware-prh}

cd "$(git rev-parse --show-toplevel)"
git ls-files -z --cached --others --exclude-standard \
  | tar --null -T - -cf - \
  | ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$IP" \
      "mkdir -p $REMOTE_DIR && tar --no-same-owner -xf - -C $REMOTE_DIR"
echo "code synced to $IP:$REMOTE_DIR"
