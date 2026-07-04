#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPYCACHEPREFIX="${ROOT_DIR}/.pj3_work/pycache"

if [ ! -f target/weights/model.pt ]; then
  "${PYTHON_BIN}" scripts/generate_toy_weights.py \
    --config target/model_config.json \
    --output target/weights/model.pt
fi

"${PYTHON_BIN}" evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto

"${PYTHON_BIN}" evaluator/stress_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto

"${PYTHON_BIN}" evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto

"${PYTHON_BIN}" evaluator/benchmark_mixed.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
