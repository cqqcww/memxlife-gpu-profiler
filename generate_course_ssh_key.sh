#!/usr/bin/env bash

set -euo pipefail

KEY_BASE="${COURSE_SSH_KEY_BASE:-$HOME/.ssh/mlsys_course_ed25519}"

mkdir -p "$(dirname "${KEY_BASE}")"
chmod 700 "$(dirname "${KEY_BASE}")"

if [ -f "${KEY_BASE}" ] || [ -f "${KEY_BASE}.pub" ]; then
  echo "Key already exists at ${KEY_BASE}"
  exit 0
fi

ssh-keygen -t ed25519 -N "" -C "23302010089@mlsys-course" -f "${KEY_BASE}"
chmod 600 "${KEY_BASE}"
chmod 644 "${KEY_BASE}.pub"
echo "Generated key pair at ${KEY_BASE}"
