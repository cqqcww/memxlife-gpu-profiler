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
import re
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
output_id = payload.get("output_file", "")
if not output_id:
    message = payload.get("message", "")
    match = re.search(r"Mission id:\s*([0-9a-fA-F]+)", message)
    if match:
        output_id = match.group(1)
if output_id:
    out_path.write_text(output_id + "\n", encoding="utf-8")
    print(f"Saved output id to {out_path}")
else:
    print("No output id found in submit response", file=sys.stderr)
PY
