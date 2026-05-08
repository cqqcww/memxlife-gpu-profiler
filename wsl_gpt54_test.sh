#!/bin/bash
set -e
export PATH=/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}

PROJ=/mnt/v/code_Temp/memxlife-gpu-profiler/memxlife-project
cd "$PROJ"

# Install openai properly
pip3 install --break-system-packages openai -q 2>/dev/null || \
pip3 install openai -q 2>/dev/null || \
sudo pip3 install --break-system-packages openai -q 2>/dev/null || true

python3 -c "import openai; print('openai OK:', openai.__version__)"

rm -rf runs/

echo "============================================"
echo "WSL + GPT-5.4 Real GPU Pipeline"
echo "============================================"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
nvcc --version 2>/dev/null | grep release
echo "============================================"

python3 main.py tests/full_target_spec.json -o runs -v 2>&1
