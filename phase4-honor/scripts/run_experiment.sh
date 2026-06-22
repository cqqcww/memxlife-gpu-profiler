#!/usr/bin/env bash

set -euo pipefail

CONFIG="${1:-configs/debug.yaml}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 agent/runner.py --config "${CONFIG}"
