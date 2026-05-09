#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${ROOT_DIR}/.phase2_work/pycache"

python3 -m phase2_agent.run_agent
