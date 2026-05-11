#!/usr/bin/env bash

set -euo pipefail

SSH_PORT="${1:?usage: course_remote_exec.sh <ssh_port> [server] <remote_command>}"
shift

SERVER="10.176.37.31"
if [ "$#" -ge 2 ]; then
  SERVER="$1"
  shift
fi

REMOTE_CMD="${1:?usage: course_remote_exec.sh <ssh_port> [server] <remote_command>}"
SSH_KEY="${COURSE_SSH_KEY:-/Users/amanda/.ssh/mlsys_course_ed25519}"
SSH_PASSWORD="${COURSE_SSH_PASSWORD:-mlsys}"
REMOTE_CMD_B64="$(printf '%s' "${REMOTE_CMD}" | base64 | tr -d '\n')"
COURSE_REMOTE_WRAPPER="python3 -c 'import base64,subprocess; cmd=base64.b64decode(\"${REMOTE_CMD_B64}\").decode(); raise SystemExit(subprocess.call([\"bash\",\"-lc\",cmd]))'"
export COURSE_REMOTE_WRAPPER

run_with_key() {
  ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" -p "${SSH_PORT}" "root@${SERVER}" "${COURSE_REMOTE_WRAPPER}"
}

run_with_password() {
  export COURSE_REMOTE_SERVER="${SERVER}"
  export COURSE_REMOTE_SSH_PORT="${SSH_PORT}"
  export COURSE_REMOTE_PASSWORD="${SSH_PASSWORD}"
  expect <<'EOF'
set timeout -1
spawn ssh -o StrictHostKeyChecking=no -p $env(COURSE_REMOTE_SSH_PORT) root@$env(COURSE_REMOTE_SERVER) $env(COURSE_REMOTE_WRAPPER)
expect {
  -re "yes/no" {
    send "yes\r"
    exp_continue
  }
  -re "[Pp]assword:" {
    send "$env(COURSE_REMOTE_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
set exit_status [lindex \$result 3]
exit \$exit_status
EOF
}

if ! run_with_key; then
  echo "Key-based remote exec failed; falling back to password auth." >&2
  run_with_password
fi
