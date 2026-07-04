#!/usr/bin/env bash

set -euo pipefail

SSH_PORT="${1:?usage: course_fetch_remote_file.sh <ssh_port> <remote_path> <local_path> [server]}"
REMOTE_PATH="${2:?usage: course_fetch_remote_file.sh <ssh_port> <remote_path> <local_path> [server]}"
LOCAL_PATH="${3:?usage: course_fetch_remote_file.sh <ssh_port> <remote_path> <local_path> [server]}"
SERVER="${4:-10.176.37.31}"
SSH_KEY="${COURSE_SSH_KEY:-/Users/amanda/.ssh/mlsys_course_ed25519}"
SSH_PASSWORD="${COURSE_SSH_PASSWORD:-mlsys}"

mkdir -p "$(dirname "${LOCAL_PATH}")"

fetch_with_key() {
  scp -P "${SSH_PORT}" \
    -o StrictHostKeyChecking=no \
    -i "${SSH_KEY}" \
    "root@${SERVER}:${REMOTE_PATH}" \
    "${LOCAL_PATH}"
}

fetch_with_password() {
  export COURSE_FETCH_SERVER="${SERVER}"
  export COURSE_FETCH_SSH_PORT="${SSH_PORT}"
  export COURSE_FETCH_PASSWORD="${SSH_PASSWORD}"
  export COURSE_FETCH_REMOTE_PATH="${REMOTE_PATH}"
  export COURSE_FETCH_LOCAL_PATH="${LOCAL_PATH}"

  expect <<'EOF'
set timeout -1
spawn scp -P $env(COURSE_FETCH_SSH_PORT) -o StrictHostKeyChecking=no root@$env(COURSE_FETCH_SERVER):$env(COURSE_FETCH_REMOTE_PATH) $env(COURSE_FETCH_LOCAL_PATH)
expect {
  -re {yes/no} {
    send "yes\r"
    exp_continue
  }
  -re {[Pp]assword:} {
    send "$env(COURSE_FETCH_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
set exit_status [lindex $result 3]
exit $exit_status
EOF
}

if ! fetch_with_key; then
  echo "Key-based remote fetch failed; falling back to password auth." >&2
  fetch_with_password
fi

echo "Fetched ${REMOTE_PATH} to ${LOCAL_PATH}"
