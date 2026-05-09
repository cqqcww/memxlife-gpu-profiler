#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
USER_ID="${COURSE_USER_ID:-23302010089}"

curl -X POST "http://${SERVER}:8080/finish" \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"${USER_ID}\"}"
echo
