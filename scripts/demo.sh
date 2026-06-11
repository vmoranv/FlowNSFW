#!/bin/bash
# FlowNSFW 一键演示脚本

set -e

echo "=== FlowNSFW 演示 ==="
echo ""

# 检查权重文件
if [ ! -f "final.pt" ]; then
    echo "❌ 权重文件 final.pt 未找到"
    echo "请确保在 output.zip 解压后的根目录运行此脚本"
    exit 1
fi

# 检查 manifest
if [ ! -f "datasets/manifest_v4_clean_wsl.json" ]; then
    echo "❌ 测试集 manifest 未找到"
    exit 1
fi

echo "📊 运行推理（224 测试视频）..."
echo ""

# 运行推理
python3 scripts/infer.py \
    --ckpt final.pt \
    --manifest datasets/manifest_v4_clean_wsl.json \
    --output demo_results.json \
    --clip-len 8 \
    --stride 4 \
    --device cuda 2>&1 | tee demo_output.log

echo ""
echo "✅ 演示完成！"
echo ""
echo "结果文件："
echo "  demo_results.json    — 完整分类结果"
echo "  demo_output.log      — 推理日志"
echo ""
echo "查看报告："
echo "  cat BENCHMARK.md"
