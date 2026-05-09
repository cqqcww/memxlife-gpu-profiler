#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
USER_ID="${COURSE_USER_ID:-23302010089}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_JSON="$(mktemp)"

curl -X POST "http://${SERVER}:8080/submit2" \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"${USER_ID}\",\"gpu\":1}" | tee "${TMP_JSON}"
echo

python3 - "${TMP_JSON}" "${REPO_ROOT}/output_id2.txt" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
output_file = payload.get("output_file", "")
if output_file:
    out_path.write_text(output_file + "\n", encoding="utf-8")
    print(f"Saved output id to {out_path}")
else:
    print("No output_file found in submit response", file=sys.stderr)
PY
