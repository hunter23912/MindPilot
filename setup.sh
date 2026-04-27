#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# Optimized version: all packages installed via pip in correct dependency order
# PyTorch installed first to avoid automatic version conflicts.

set -e

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Please install Miniconda/Anaconda first, then rerun this script."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

# Remove old incomplete environment if it exists (optional but recommended)
if conda env list | awk '{print $1}' | grep -qx "MindPilot"; then
    echo "Removing existing incomplete 'MindPilot' environment..."
    conda env remove -n MindPilot -y
fi

echo "Creating new conda environment 'MindPilot' with Python 3.10.8..."
conda create -n MindPilot python=3.10.8 -y

conda activate MindPilot

# Ensure pip is up-to-date
pip install --upgrade pip

# ------------------------------------------------------------
# 1. Install PyTorch first (core dependency for everything else)
# ------------------------------------------------------------
pip install torch==2.11.0 torchvision==0.26.0

# ------------------------------------------------------------
# 2. Scientific computing and visualization
# ------------------------------------------------------------
pip install numpy==2.2.6 matplotlib tqdm scikit-image jupyterlab seaborn

# ------------------------------------------------------------
# 3. Acceleration library
# ------------------------------------------------------------
pip install accelerate==1.13.0

# ------------------------------------------------------------
# 4. CLIP / DALL-E / OpenCLIP related
# ------------------------------------------------------------
pip install clip-retrieval clip pandas ftfy regex kornia umap-learn
pip install dalle2-pytorch
pip install open_clip_torch==2.32.0

# ------------------------------------------------------------
# 5. Transformers ecosystem (specific versions)
# ------------------------------------------------------------
pip install transformers==5.6.2
pip install diffusers==0.37.1

# ------------------------------------------------------------
# 6. Neuroscience / MNE / BrainDecode
# ------------------------------------------------------------
pip install braindecode==0.8.1
pip install mne

# ------------------------------------------------------------
# 7. Other utilities
# ------------------------------------------------------------
pip install info-nce-pytorch==0.1.0
pip install pytorch-msssim
pip install reformer_pytorch
pip install wandb einops==0.8.2
pip install natsort
pip install gpytorch
pip install hf_transfer
pip install huggingface_hub==1.12.0
pip install mne

# Fix libstdc++ ABI for scipy 1.15+ (add conda env lib to LD_LIBRARY_PATH)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'HOOKEOF'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
HOOKEOF

echo ""
echo "Environment 'MindPilot' created successfully!"
echo "To activate the environment, run: conda activate MindPilot"

# download test images
# HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_FILE_LOCKING=1 hf download gasparyanartur/things-eeg2 --repo-type dataset --local-dir ./data/things-eeg2 --include "imgs/test_images.zip"