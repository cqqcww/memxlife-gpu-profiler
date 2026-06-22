#!/usr/bin/env bash

set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${WORKFLOW_DIR}/.." && pwd)"
STATE_DIR="${WORKFLOW_DIR}/state"
LOG_DIR="${WORKFLOW_DIR}/logs"
ARTIFACT_DIR="${WORKFLOW_DIR}/artifacts"

if [ -f "${WORKFLOW_DIR}/config.env" ]; then
  # shellcheck disable=SC1091
  source "${WORKFLOW_DIR}/config.env"
fi

COURSE_SERVER="${COURSE_SERVER:-10.176.37.31}"
COURSE_USER_ID="${COURSE_USER_ID:-23302010089}"
COURSE_GPU="${COURSE_GPU:-1}"
REMOTE_DIR="${REMOTE_DIR:-/workspace}"
COURSE_SSH_KEY="${COURSE_SSH_KEY:-/Users/amanda/.ssh/mlsys_course_ed25519}"
COURSE_SSH_PASSWORD="${COURSE_SSH_PASSWORD:-mlsys}"
COURSE_SUBMIT_PATH="${COURSE_SUBMIT_PATH:-/submit4}"
COURSE_START_TIMEOUT_SECONDS="${COURSE_START_TIMEOUT_SECONDS:-1800}"
COURSE_START_POLL_SECONDS="${COURSE_START_POLL_SECONDS:-10}"
COURSE_CURL_CONNECT_TIMEOUT="${COURSE_CURL_CONNECT_TIMEOUT:-8}"
COURSE_CURL_MAX_TIME="${COURSE_CURL_MAX_TIME:-30}"
PHASE4_FETCH_MAX_BYTES="${PHASE4_FETCH_MAX_BYTES:-25000000}"
REMOTE_TAIL_LINES="${REMOTE_TAIL_LINES:-120}"

SSH_PORT_FILE="${STATE_DIR}/ssh_port.txt"
DEV_MISSION_FILE="${STATE_DIR}/dev_mission_id.txt"
OFFICIAL_ID_FILE="${STATE_DIR}/official_output_id.txt"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${ARTIFACT_DIR}"

log() {
  printf '[phase4-selfcmd] %s\n' "$*" >&2
}

die() {
  printf '[phase4-selfcmd:error] %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

curl_course() {
  curl --connect-timeout "${COURSE_CURL_CONNECT_TIMEOUT}" \
    --max-time "${COURSE_CURL_MAX_TIME}" \
    -sS "$@"
}

current_ssh_port() {
  if [ -n "${COURSE_SSH_PORT:-}" ]; then
    printf '%s\n' "${COURSE_SSH_PORT}"
    return
  fi
  if [ -f "${SSH_PORT_FILE}" ]; then
    tr -d '\n' < "${SSH_PORT_FILE}"
    return
  fi
  die "No ssh port known. Run './selfcmd start' first, or set COURSE_SSH_PORT."
}

remote_exec_on_port() {
  local ssh_port="$1"
  shift
  local remote_cmd="$*"
  local remote_cmd_b64
  remote_cmd_b64="$(printf '%s' "${remote_cmd}" | base64 | tr -d '\n')"
  local wrapper
  wrapper="python3 -c 'import base64,subprocess; cmd=base64.b64decode(\"${remote_cmd_b64}\").decode(); raise SystemExit(subprocess.call([\"bash\",\"-lc\",cmd]))'"

  if ssh -o StrictHostKeyChecking=no -i "${COURSE_SSH_KEY}" -p "${ssh_port}" \
    "root@${COURSE_SERVER}" "${wrapper}"; then
    return 0
  fi

  need_cmd expect
  log "Key-based ssh failed; falling back to password auth."
  export PHASE4_REMOTE_SERVER="${COURSE_SERVER}"
  export PHASE4_REMOTE_SSH_PORT="${ssh_port}"
  export PHASE4_REMOTE_PASSWORD="${COURSE_SSH_PASSWORD}"
  export PHASE4_REMOTE_WRAPPER="${wrapper}"
  expect <<'EOF'
set timeout -1
spawn ssh -o StrictHostKeyChecking=no -p $env(PHASE4_REMOTE_SSH_PORT) root@$env(PHASE4_REMOTE_SERVER) $env(PHASE4_REMOTE_WRAPPER)
expect {
  "*yes/no*" {
    send "yes\r"
    exp_continue
  }
  "*password:*" {
    send "$env(PHASE4_REMOTE_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
set exit_status [lindex $result 3]
exit $exit_status
EOF
}

remote_exec() {
  local ssh_port
  ssh_port="$(current_ssh_port)"
  remote_exec_on_port "${ssh_port}" "$@"
}

copy_to_remote() {
  local local_path="$1"
  local remote_path="$2"
  local ssh_port
  ssh_port="$(current_ssh_port)"

  if scp -P "${ssh_port}" -o StrictHostKeyChecking=no -i "${COURSE_SSH_KEY}" \
    "${local_path}" "root@${COURSE_SERVER}:${remote_path}"; then
    return 0
  fi

  need_cmd expect
  log "Key-based scp failed; falling back to password auth."
  export PHASE4_SCP_SERVER="${COURSE_SERVER}"
  export PHASE4_SCP_SSH_PORT="${ssh_port}"
  export PHASE4_SCP_PASSWORD="${COURSE_SSH_PASSWORD}"
  export PHASE4_SCP_LOCAL="${local_path}"
  export PHASE4_SCP_REMOTE="${remote_path}"
  expect <<'EOF'
set timeout -1
spawn scp -P $env(PHASE4_SCP_SSH_PORT) -o StrictHostKeyChecking=no $env(PHASE4_SCP_LOCAL) root@$env(PHASE4_SCP_SERVER):$env(PHASE4_SCP_REMOTE)
expect {
  "*yes/no*" {
    send "yes\r"
    exp_continue
  }
  "*password:*" {
    send "$env(PHASE4_SCP_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
set exit_status [lindex $result 3]
exit $exit_status
EOF
}

copy_from_remote() {
  local remote_path="$1"
  local local_path="$2"
  local ssh_port
  ssh_port="$(current_ssh_port)"

  if scp -P "${ssh_port}" -o StrictHostKeyChecking=no -i "${COURSE_SSH_KEY}" \
    "root@${COURSE_SERVER}:${remote_path}" "${local_path}"; then
    return 0
  fi

  need_cmd expect
  log "Key-based scp fetch failed; falling back to password auth."
  export PHASE4_SCP_SERVER="${COURSE_SERVER}"
  export PHASE4_SCP_SSH_PORT="${ssh_port}"
  export PHASE4_SCP_PASSWORD="${COURSE_SSH_PASSWORD}"
  export PHASE4_SCP_REMOTE="${remote_path}"
  export PHASE4_SCP_LOCAL="${local_path}"
  expect <<'EOF'
set timeout -1
spawn scp -P $env(PHASE4_SCP_SSH_PORT) -o StrictHostKeyChecking=no root@$env(PHASE4_SCP_SERVER):$env(PHASE4_SCP_REMOTE) $env(PHASE4_SCP_LOCAL)
expect {
  "*yes/no*" {
    send "yes\r"
    exp_continue
  }
  "*password:*" {
    send "$env(PHASE4_SCP_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
set exit_status [lindex $result 3]
exit $exit_status
EOF
}

make_source_archive() {
  local out_tar="$1"
  COPYFILE_DISABLE=1 tar --format ustar -czf "${out_tar}" \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude "*/__pycache__" \
    --exclude ".pytest_cache" \
    --exclude "*/.pytest_cache" \
    --exclude ".DS_Store" \
    --exclude "._*" \
    --exclude "*/._*" \
    --exclude ".AppleDouble" \
    --exclude "*/.AppleDouble" \
    --exclude "runs" \
    --exclude "data_cache" \
    --exclude "checkpoints" \
    --exclude ".cache" \
    --exclude ".hf_cache" \
    --exclude "selfcmd-workflow/state" \
    --exclude "selfcmd-workflow/logs" \
    --exclude "selfcmd-workflow/artifacts" \
    --exclude "*.pt" \
    --exclude "*.pth" \
    -C "${PROJECT_ROOT}" .
}

sync_source_overlay() {
  local tmp_tar
  local remote_tar
  tmp_tar="$(mktemp /tmp/phase4-source.XXXXXX).tar.gz"
  remote_tar="/tmp/$(basename "${tmp_tar}")"
  make_source_archive "${tmp_tar}"
  copy_to_remote "${tmp_tar}" "${remote_tar}"
  remote_exec "mkdir -p '${REMOTE_DIR}' && tar -xzf '${remote_tar}' -C '${REMOTE_DIR}' && rm -f '${remote_tar}'"
  rm -f "${tmp_tar}"
  log "Synced source overlay to ${COURSE_SERVER}:${REMOTE_DIR}"
}

clean_remote_source() {
  remote_exec "cd '${REMOTE_DIR}' && find . -name '._*' -delete && rm -rf training_framework agent configs tests scripts selfcmd-workflow fixtures train.py report4.md README.md requirements.txt pyproject.toml setup.cfg setup.py __pycache__ .pytest_cache .phase2_work evaluator target memxlife-examples-static-phase1 memxlife-origin-static memxlife-project global-project-plan phase2_agent phase3_engine_sources stage2_outputs stage3_outputs course_*.sh generate_course_ssh_key.sh setup_course_ssh_aliases.sh optimized_lora.cu output.md output_id2.txt output_id3.txt phase2_optimization_journey.md report2.md report3.md result.log results.log run.sh ta_core_bundle_20260514 ta_core_bundle_20260514.tar.gz tmp_phase4_update workspace wsl_*.sh"
  log "Removed known source/noise paths on remote while preserving runs/data_cache/checkpoints."
}

run_logged_remote() {
  local label="$1"
  shift
  local cmd="$*"
  local safe_label
  safe_label="$(printf '%s' "${label}" | tr -c 'A-Za-z0-9_.-' '_')"
  local quoted_cmd
  printf -v quoted_cmd '%q' "${cmd}"
  remote_exec "cd '${REMOTE_DIR}' && mkdir -p selfcmd-workflow/logs && LOG='selfcmd-workflow/logs/${safe_label}.log' && set +e; bash -lc ${quoted_cmd} > \"\${LOG}\" 2>&1; status=\$?; echo remote_log:${REMOTE_DIR}/\${LOG}; tail -n ${REMOTE_TAIL_LINES} \"\${LOG}\"; exit \${status}"
}

pack_remote_artifacts() {
  local remote_tar="$1"
  remote_exec "cd '${REMOTE_DIR}' && PHASE4_REMOTE_TAR='${remote_tar}' PHASE4_FETCH_MAX_BYTES='${PHASE4_FETCH_MAX_BYTES}' python3 - <<'PY'
import os
import tarfile
from pathlib import Path

root = Path('.').resolve()
out = Path(os.environ['PHASE4_REMOTE_TAR'])
max_bytes = int(os.environ.get('PHASE4_FETCH_MAX_BYTES', '25000000'))
patterns = [
    'report4.md',
    'README.md',
    'configs/*.yaml',
    'configs/base/*.yaml',
    'configs/model_profiles/*.yaml',
    'configs/data_profiles/*.yaml',
    'configs/matrices/*.yaml',
    'configs/auto_probes/*.yaml',
    'runs/*/summary.json',
    'runs/*/summary.md',
    'runs/*/preflight.json',
    'runs/*/preflight.md',
    'runs/*/agent_summary.json',
    'runs/*/agent_summary.md',
    'runs/*/agent_patch_proposal.md',
    'runs/*/events.jsonl',
    'runs/*/*.log',
    'runs/*/copied_config.yaml',
    'runs/*/config.yaml',
    'runs/matrix_configs/*.json',
    'runs/stability_configs/*.json',
    'runs/matrix_summaries/*.json',
    'runs/matrix_summaries/*.md',
    'runs/auto_probe_summaries/*.json',
    'runs/auto_probe_summaries/*.md',
    'runs/stability_summaries/*.json',
    'runs/stability_summaries/*.md',
    'runs/recommendations/*.json',
    'runs/recommendations/*.md',
    'runs/evidence_table.md',
    'runs/evidence_table.csv',
    'selfcmd-workflow/logs/*.log',
]
added = []
with tarfile.open(out, 'w:gz') as tar:
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                continue
            arcname = path.relative_to(root)
            if str(arcname) in added:
                continue
            tar.add(path, arcname=str(arcname))
            added.append(str(arcname))
print('packed_artifacts=' + str(len(added)))
for item in added:
    print(item)
PY"
}
