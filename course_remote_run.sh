#!/usr/bin/env bash

set -euo pipefail

SSH_PORT="${1:?usage: course_remote_run.sh <ssh_port> [server] [remote_dir]}"
SERVER="${2:-10.176.37.31}"
REMOTE_DIR="${3:-/workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${REPO_ROOT}/course_remote_exec.sh" "${SSH_PORT}" "${SERVER}" \
  "cd ${REMOTE_DIR} && chmod +x run.sh && ./run.sh"
