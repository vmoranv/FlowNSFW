#!/bin/bash
# A10 一键安装脚本

set -e

echo "=========================================="
echo "FlowNSFW A10 环境安装"
echo "=========================================="

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA driver not found"
    exit 1
fi

echo "✓ NVIDIA driver detected"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python $PYTHON_VERSION detected"

# 安装 PyTorch (CUDA 12.1)
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 Mamba SSM
echo ""
echo "Installing Mamba SSM (CUDA kernels)..."
pip install causal-conv1d mamba-ssm --extra-index-url https://pypi.nvidia.com

# 安装项目
echo ""
echo "Installing FlowNSFW..."
pip install -e .

# 安装其他依赖
echo ""
echo "Installing additional dependencies..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

try:
    import mamba_ssm
    print(f'Mamba SSM: OK')
except:
    print(f'Mamba SSM: NOT FOUND (will fallback to PyTorch)')

import cv2
print(f'OpenCV: {cv2.__version__}')

from flow_nsfw import FlowNSFW
print(f'FlowNSFW: OK')
"

echo ""
echo "=========================================="
echo "✅ Installation Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset manifest"
echo "  2. Run training: bash scripts/train_a10_single.sh"
echo "  3. Or run full ablation: bash scripts/train_a10_full.sh"
