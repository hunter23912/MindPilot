#!/bin/bash
export PROJECT_ROOT="/root/autodl-tmp/MindPilot"
set -a # 自动导出
source "${PROJECT_ROOT}/mindpilot.env"
set +a # 关闭自动导出变量

# Set library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 把model目录和项目根目录加入PYTHONPATH，避免重复把根目录塞进 env 文件
export PYTHONPATH=${PROJECT_ROOT}/model:${PROJECT_ROOT}:${PYTHONPATH:-}

source /etc/network_turbo
# 强制走 hf-mirror，避免直连 huggingface.co 503
export HF_ENDPOINT=https://hf-mirror.com
# 减少显存碎片，降低 OOM 概率
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1. 先做语法检查（仅检查，不运行）
echo "正在检查 Python 语法..."
CUDA_VISIBLE_DEVICES=0 python -m py_compile ${PROJECT_ROOT}/experiments/benchmark_offline_generation.py

# 2. 检查通过 → 运行脚本；检查失败 → 终止
echo "语法检查通过，开始运行脚本..."
CUDA_VISIBLE_DEVICES=0 python ${PROJECT_ROOT}/experiments/benchmark_offline_generation.py