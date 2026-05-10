#!/usr/bin/env bash

set -euo pipefail

SSH_PORT="${1:?usage: course_remote_run.sh <ssh_port> [server] [remote_dir]}"
SERVER="${2:-10.176.37.31}"
REMOTE_DIR="${3:-/workspace}"
SSH_KEY="${COURSE_SSH_KEY:-/Users/amanda/.ssh/mlsys_course_ed25519}"

ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" -p "${SSH_PORT}" "root@${SERVER}" \
  "cd ${REMOTE_DIR} && chmod +x run.sh && ./run.sh"
