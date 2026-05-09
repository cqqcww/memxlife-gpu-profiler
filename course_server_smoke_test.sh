#!/usr/bin/env bash

set -euo pipefail

USER_NAME="${COURSE_SSH_USER:-23302010089}"
TARGET="${1:-gpu1}"

case "$TARGET" in
  gpu1)
    HOST="10.190.248.247"
    PORT="59815"
    ;;
  gpu2)
    HOST="10.193.2.99"
    PORT="30148"
    ;;
  *)
    echo "Usage: $0 [gpu1|gpu2]" >&2
    exit 2
    ;;
esac

echo "Connecting to ${USER_NAME}@${HOST}:${PORT}"
ssh -tt \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -p "${PORT}" \
  "${USER_NAME}@${HOST}" \
  'echo "HOST=$(hostname)"; echo "USER=$(whoami)"; python3 -V || true; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true; nvcc --version | grep release || true'
