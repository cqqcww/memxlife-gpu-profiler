#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
GPU_MODE="${2:-1}"
USER_ID="${COURSE_USER_ID:-23302010089}"

curl -X POST "http://${SERVER}:8080/start" \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"${USER_ID}\",\"gpu\":${GPU_MODE}}"
echo
