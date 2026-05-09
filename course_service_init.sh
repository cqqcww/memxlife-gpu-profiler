#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
USER_ID="${COURSE_USER_ID:-23302010089}"
KEY_PATH="${COURSE_SSH_KEY_PATH:-$HOME/.ssh/mlsys_course_ed25519.pub}"

if [ ! -f "${KEY_PATH}" ]; then
  echo "Missing public key: ${KEY_PATH}" >&2
  exit 1
fi

PUB_KEY="$(tr -d '\n' < "${KEY_PATH}")"

curl -X POST "http://${SERVER}:8080/init" \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"${USER_ID}\",\"ssh_pub_key\":\"${PUB_KEY}\"}"
echo
