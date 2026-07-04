#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
GPU_MODE="${2:-1}"
REMOTE_DIR="${3:-/workspace}"
TIMEOUT_SECONDS="${COURSE_START_TIMEOUT_SECONDS:-1800}"
POLL_SECONDS="${COURSE_START_POLL_SECONDS:-10}"
USER_ID="${COURSE_USER_ID:-23302010089}"
COURSE_CURL_CONNECT_TIMEOUT="${COURSE_CURL_CONNECT_TIMEOUT:-8}"
COURSE_CURL_MAX_TIME="${COURSE_CURL_MAX_TIME:-30}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURL_OPTS=(--connect-timeout "${COURSE_CURL_CONNECT_TIMEOUT}" --max-time "${COURSE_CURL_MAX_TIME}" -sS)

DEV_MISSION_ID=""
CLEANUP_DEV=0

cleanup() {
  if [ "${CLEANUP_DEV}" -eq 1 ]; then
    "${REPO_ROOT}/course_service_finish.sh" "${SERVER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start_payload="$(curl "${CURL_OPTS[@]}" -X POST "http://${SERVER}:8080/start" \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"${USER_ID}\",\"gpu\":${GPU_MODE}}")"

DEV_MISSION_ID="$(python3 - "${start_payload}" <<'PY'
import json
import re
import sys

payload = json.loads(sys.argv[1])
message = payload.get("message", "")
match = re.search(r"Mission id:\s*([0-9a-fA-F]+)", message)
if not match:
    match = re.search(r"mission\s+([0-9a-fA-F]+)", message, re.IGNORECASE)
if not match:
    raise SystemExit("Could not parse dev mission id from /start response")
print(match.group(1))
PY
)"

echo "Queued dev mission: ${DEV_MISSION_ID}"
CLEANUP_DEV=1

SSH_PORT=""
LAST_STATUS=""
DEADLINE=$((SECONDS + TIMEOUT_SECONDS))

while [ "${SECONDS}" -lt "${DEADLINE}" ]; do
  status_payload="$(curl "${CURL_OPTS[@]}" "http://${SERVER}:8080/submit_status/${DEV_MISSION_ID}")"
  parsed_status="$(python3 - "${status_payload}" <<'PY'
import json
import re
import sys

payload = json.loads(sys.argv[1])
message = payload.get("message", "")
status = ""
ssh_port = ""
for raw_line in message.splitlines():
    line = raw_line.strip()
    if line.startswith("status:"):
        status = line.split(":", 1)[1].strip()
    elif line.startswith("ssh_port:"):
        ssh_port = line.split(":", 1)[1].strip()
    else:
        match = re.search(r"ssh_port[:=]\s*(\d+)", line)
        if match:
            ssh_port = match.group(1)
if not ssh_port:
    match = re.search(r"ssh_port[:=]\s*(\d+)", message)
    if match:
        ssh_port = match.group(1)
print(status)
print(ssh_port)
PY
)"
  CURRENT_STATUS="$(printf '%s\n' "${parsed_status}" | sed -n '1p')"
  CURRENT_SSH_PORT="$(printf '%s\n' "${parsed_status}" | sed -n '2p')"

  if [ "${CURRENT_STATUS}" != "${LAST_STATUS}" ]; then
    echo "Dev mission status: ${CURRENT_STATUS:-unknown}"
    LAST_STATUS="${CURRENT_STATUS}"
  fi

  if [ "${CURRENT_STATUS}" = "running" ] && [ -n "${CURRENT_SSH_PORT}" ]; then
    SSH_PORT="${CURRENT_SSH_PORT}"
    break
  fi

  if [ "${CURRENT_STATUS}" = "failed" ] || [ "${CURRENT_STATUS}" = "killed" ] || [ "${CURRENT_STATUS}" = "completed" ]; then
    echo "Dev mission ended before exposing ssh_port." >&2
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done

if [ -z "${SSH_PORT}" ]; then
  echo "Timed out waiting for dev mission ssh_port." >&2
  exit 1
fi

echo "Dev mission ssh_port: ${SSH_PORT}"

"${REPO_ROOT}/course_sync_workspace.sh" "${SSH_PORT}" "${SERVER}" "${REMOTE_DIR}"

VERIFY_CMD="$(cat <<EOF
cd ${REMOTE_DIR}
test -f run.sh
test -f workspace/engine.py
python3 -m py_compile workspace/engine.py
bash run.sh
echo pj3_remote_prepare_ok
EOF
)"

echo "Remote phase3 sanity:"
"${REPO_ROOT}/course_remote_exec.sh" "${SSH_PORT}" "${SERVER}" "${VERIFY_CMD}"

"${REPO_ROOT}/course_service_finish.sh" "${SERVER}"
CLEANUP_DEV=0

echo "Submitting official Phase 3 run..."
"${REPO_ROOT}/course_submit3.sh" "${SERVER}"
OFFICIAL_MISSION_ID="$(tr -d '\n' < "${REPO_ROOT}/output_id3.txt")"
echo "Official mission id: ${OFFICIAL_MISSION_ID}"
