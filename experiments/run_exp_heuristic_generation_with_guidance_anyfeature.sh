#!/bin/bash

# Run heuristic generation (anyfeature) — simplified like offline launcher
# Usage: bash experiments/run_exp_heuristic_generation_with_guidance_anyfeature.sh

export PROJECT_ROOT="/root/autodl-tmp/MindPilot"

set -a
source "${PROJECT_ROOT}/mindpilot.env"
set +a

# library path + pythonpath (keep same style as other run_*.sh)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${PROJECT_ROOT}/model:${PROJECT_ROOT}:${PYTHONPATH:-}

source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_ID=0

TARGET_MODULE="${PROJECT_ROOT}/experiments/exp-heuristic_generation_with_guidance_anyfeature.py"

echo "Starting Heuristic Generation (anyfeature)"
echo "Project root: ${PROJECT_ROOT}"
echo "GPU device:   ${GPU_ID}"

echo "正在检查 Python 语法..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m py_compile "${TARGET_MODULE}"

echo "语法检查通过，开始运行脚本..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python "${TARGET_MODULE}"
