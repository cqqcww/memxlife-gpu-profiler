#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ID="${2:-$(tr -d '\n' < "${REPO_ROOT}/output_id2.txt")}"

if [ -z "${OUTPUT_ID}" ]; then
  echo "Missing output id" >&2
  exit 1
fi

curl "http://${SERVER}:8080/submit_status/${OUTPUT_ID}"
echo
