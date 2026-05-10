#!/usr/bin/env bash

set -euo pipefail

SSH_PORT="${1:?usage: course_sync_workspace.sh <ssh_port> [server] [remote_dir]}"
SERVER="${2:-10.176.37.31}"
REMOTE_DIR="${3:-/workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_KEY="${COURSE_SSH_KEY:-/Users/amanda/.ssh/mlsys_course_ed25519}"
REMOTE="root@${SERVER}:${REMOTE_DIR}/"

rsync -az \
  --exclude ".git/" \
  --exclude ".phase2_work/builds/" \
  --exclude ".phase2_work/candidates/" \
  --exclude "__pycache__/" \
  --exclude ".DS_Store" \
  -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p ${SSH_PORT}" \
  "${REPO_ROOT}/" \
  "${REMOTE}"

echo "Synced ${REPO_ROOT} to ${REMOTE}"
