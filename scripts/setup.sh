#!/bin/bash
# FlowNSFW 一键安装脚本

set -e

echo "=== FlowNSFW 环境配置 ==="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装"
    exit 1
fi

PYTHON_VER=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
echo "✅ Python $PYTHON_VER"

# 检查 CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "✅ CUDA $CUDA_VER"
else
    echo "⚠️  CUDA 未检测到，将使用 CPU（推理速度较慢）"
fi

# 安装 PyTorch
echo ""
echo "[1/3] 安装 PyTorch..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# 安装基础依赖
echo "[2/3] 安装基础依赖..."
pip3 install opencv-python numpy pillow -q

# 安装 Mamba SSM（可选）
echo "[3/3] 安装 Mamba SSM（可选，失败时自动降级）..."
pip3 install mamba-ssm 2>/dev/null || echo "⚠️  Mamba SSM 安装失败，将使用 PyTorch 原生实现"

echo ""
echo "✅ 环境配置完成！"
echo ""
echo "运行演示："
echo "  bash scripts/demo.sh"
