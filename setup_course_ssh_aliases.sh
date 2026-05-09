#!/usr/bin/env bash

set -euo pipefail

USER_NAME="${COURSE_SSH_USER:-23302010089}"
SSH_DIR="${HOME}/.ssh"
CONFIG_PATH="${SSH_DIR}/config"

mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"
touch "${CONFIG_PATH}"
chmod 600 "${CONFIG_PATH}"

TMP_PATH="$(mktemp)"
cp "${CONFIG_PATH}" "${TMP_PATH}"

python3 - "${CONFIG_PATH}" "${USER_NAME}" <<'PY'
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
user_name = sys.argv[2]

entries = {
    "mlsys-gpu1": {
        "HostName": "10.190.248.247",
        "Port": "59815",
        "User": user_name,
    },
    "mlsys-gpu2": {
        "HostName": "10.193.2.99",
        "Port": "30148",
        "User": user_name,
    },
    "mlsys-submit": {
        "HostName": "10.176.37.31",
        "Port": "22",
        "User": user_name,
    },
}

existing = config_path.read_text(encoding="utf-8")
blocks = []
for host, values in entries.items():
    block = [
        f"Host {host}",
        f"  HostName {values['HostName']}",
        f"  Port {values['Port']}",
        f"  User {values['User']}",
        "  ServerAliveInterval 30",
        "  ServerAliveCountMax 3",
        "  StrictHostKeyChecking accept-new",
        "",
    ]
    marker = f"Host {host}\n"
    if marker in existing:
        continue
    blocks.append("\n".join(block))

if blocks:
    if existing and not existing.endswith("\n"):
        existing += "\n"
    existing += "\n".join(blocks)
    config_path.write_text(existing, encoding="utf-8")
PY

echo "SSH aliases configured in ${CONFIG_PATH}"
echo "You can now use: ssh mlsys-gpu1"
echo "Or run: ./course_server_smoke_test.sh gpu1"
