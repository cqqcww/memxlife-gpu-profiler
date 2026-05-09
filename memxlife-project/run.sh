#!/bin/bash
set -e

# ── Environment setup ─────────────────────────────────────────
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Install dependencies ──────────────────────────────────────
pip3 install openai -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --default-timeout 10 -q 2>/dev/null || \
pip3 install openai --default-timeout 10 -q 2>/dev/null || true

# ── Determine target spec location ────────────────────────────
TARGET_SPEC="/target/target_spec.json"
if [ ! -f "$TARGET_SPEC" ]; then
    TARGET_SPEC="$SCRIPT_DIR/tests/full_target_spec.json"
fi

echo "============================================"
echo "MemXLife GPU Profiling Agent System"
echo "  Student: 23302010089"
echo "============================================"
echo "Target spec: $TARGET_SPEC"
echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,clocks.current.graphics --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo "CUDA:"
nvcc --version 2>/dev/null | grep "release" || echo "  nvcc not available"
echo "API_KEY set: $([ -n \"$API_KEY\" ] && echo YES || echo NO)"
echo "BASE_MODEL: ${BASE_MODEL:-not set}"
echo "BASE_URL: ${BASE_URL:-not set}"
echo "============================================"

# ── Run the agent ─────────────────────────────────────────────
python3 main.py "$TARGET_SPEC" -o runs -v 2>&1 | tee /workspace/results.log 2>/dev/null || \
python3 main.py "$TARGET_SPEC" -o runs -v

echo ""
echo "============================================"
echo "Agent run complete."
echo "============================================"
