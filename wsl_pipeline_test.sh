#!/bin/bash
set -e
export PATH=/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}

PROJ=/mnt/v/code_Temp/memxlife-gpu-profiler/memxlife-project
cd "$PROJ"

# Install deps with sudo
sudo pip3 install anthropic openai -q 2>/dev/null || sudo pip install anthropic openai -q 2>/dev/null || {
    echo "pip install failed, trying with --break-system-packages"
    sudo pip3 install --break-system-packages anthropic openai -q 2>/dev/null || true
}

# Verify
python3 -c "import anthropic; print('anthropic OK:', anthropic.__version__)" 2>&1

# Set API key for dev
export ANTHROPIC_API_KEY="sk-P6IbxZEFR7qH7Nzx2f1HMQjr3lCnPpyOCeuWxfoew8zJMlUK"

# Clean previous runs
rm -rf runs/

echo "============================================"
echo "WSL Full Pipeline Test"
echo "============================================"
nvidia-smi --query-gpu=name --format=csv,noheader
nvcc --version | grep release
echo "============================================"

# Run
python3 main.py tests/full_target_spec.json -o runs -v 2>&1
