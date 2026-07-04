#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p workspace
LOG_PATH="${ROOT_DIR}/workspace/output3.log"
LEGACY_LOG_PATH="${ROOT_DIR}/workspace/results.log"
export PYTHONPYCACHEPREFIX="${ROOT_DIR}/.pj3_work/pycache"
ENGINE_RENDERER="${ROOT_DIR}/scripts/render_phase3_engine.py"
ENGINE_VARIANT="${PHASE3_ENGINE_VARIANT:-current_best}"

{
  echo "[pj3] prepare started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "[pj3] python: ${PYTHON:-python3}"
  echo "[pj3] pycache: ${PYTHONPYCACHEPREFIX}"
  echo "[pj3] engine renderer: ${ENGINE_RENDERER}"
  echo "[pj3] engine variant: ${ENGINE_VARIANT}"
  "${PYTHON:-python3}" "${ENGINE_RENDERER}" --root "${ROOT_DIR}" --variant "${ENGINE_VARIANT}"
  "${PYTHON:-python3}" -m py_compile workspace/engine.py
  echo "[pj3] engine.py py_compile passed"
  echo "[pj3] prepare finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee "${LOG_PATH}" "${LEGACY_LOG_PATH}"
