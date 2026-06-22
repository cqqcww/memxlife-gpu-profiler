#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m py_compile train.py training_framework/*.py agent/*.py
python3 train.py --config configs/debug.yaml
