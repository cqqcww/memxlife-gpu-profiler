#!/usr/bin/env bash

set -euo pipefail

SSH_PORT="${1:?usage: course_sync_workspace.sh <ssh_port> [server] [remote_dir]}"
SERVER="${2:-10.176.37.31}"
REMOTE_DIR="${3:-/workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_KEY="${COURSE_SSH_KEY:-/Users/amanda/.ssh/mlsys_course_ed25519}"
SSH_PASSWORD="${COURSE_SSH_PASSWORD:-mlsys}"
REMOTE="root@${SERVER}:${REMOTE_DIR}/"

sync_with_key() {
  rsync -az \
    --exclude ".git/" \
    --exclude ".phase2_work/builds/" \
    --exclude ".phase2_work/candidates/" \
    --exclude "__pycache__/" \
    --exclude ".DS_Store" \
    -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p ${SSH_PORT}" \
    "${REPO_ROOT}/" \
    "${REMOTE}"
}

sync_with_password() {
  local tmp_tar
  local remote_tar
  tmp_tar="$(mktemp /tmp/mlsys-course-sync.XXXXXX)"
  tmp_tar="${tmp_tar}.tar.gz"
  remote_tar="/tmp/$(basename "${tmp_tar}")"

  tar -czf "${tmp_tar}" \
    --exclude ".git" \
    --exclude ".phase2_work/builds" \
    --exclude ".phase2_work/candidates" \
    --exclude "__pycache__" \
    --exclude ".DS_Store" \
    -C "${REPO_ROOT}" .

  export COURSE_SYNC_SERVER="${SERVER}"
  export COURSE_SYNC_SSH_PORT="${SSH_PORT}"
  export COURSE_SYNC_PASSWORD="${SSH_PASSWORD}"
  export COURSE_SYNC_TAR="${tmp_tar}"
  export COURSE_SYNC_REMOTE_TAR="${remote_tar}"

  expect <<'EOF'
set timeout -1
spawn scp -P $env(COURSE_SYNC_SSH_PORT) -o StrictHostKeyChecking=no $env(COURSE_SYNC_TAR) root@$env(COURSE_SYNC_SERVER):$env(COURSE_SYNC_REMOTE_TAR)
expect {
  -re {yes/no} {
    send "yes\r"
    exp_continue
  }
  -re {[Pp]assword:} {
    send "$env(COURSE_SYNC_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
set exit_status [lindex $result 3]
exit $exit_status
EOF

  "${REPO_ROOT}/course_remote_exec.sh" "${SSH_PORT}" "${SERVER}" \
    "mkdir -p ${REMOTE_DIR} && tar -xzf ${remote_tar} -C ${REMOTE_DIR} && rm -f ${remote_tar}"

  rm -f "${tmp_tar}"
}

if ! sync_with_key; then
  echo "Key-based rsync failed; falling back to password-based archive sync." >&2
  sync_with_password
fi

echo "Synced ${REPO_ROOT} to ${REMOTE}"
