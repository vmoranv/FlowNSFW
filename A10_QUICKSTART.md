# FlowNSFW A10 快速开始

## 📦 完整部署包

**文件**: `flow-nsfw-a10-full.tar.gz` (956 MB)

**包含**:
- ✅ 完整源代码
- ✅ 训练/评测脚本
- ✅ 数据集 + manifest
- ✅ 依赖清单
- ✅ 文档

---

## 🚀 5 分钟部署

### 1. 上传到 A10
```bash
scp flow-nsfw-a10-full.tar.gz user@a10:/workspace/
```

### 2. 解压
```bash
ssh user@a10
cd /workspace
tar -xzf flow-nsfw-a10-full.tar.gz
cd flow-nsfw
```

### 3. 一键安装
```bash
bash scripts/install_a10.sh
```

这会自动安装：
- PyTorch 2.5 + CUDA 12.1
- Mamba SSM (CUDA kernel)
- OpenCV, einops, timm
- 所有依赖

### 4. 开始训练
```bash
# 推荐配置（320x320, batch=8, 30 epoch）
bash scripts/train_a10_single.sh
```

训练完成后自动评测，结果保存在 `runs/prod_mamba2_full/`

---

## 📊 预期结果

| 配置 | 分辨率 | Batch | 显存 | 训练时间 | 精度 |
|------|--------|-------|------|----------|------|
| **mamba2_full** | 320² | 8 | 16GB | ~2h (30ep) | **87-90%** |
| mamba2_full | 384² | 4 | 18GB | ~3h | 89-92% |
| mamba3_full | 320² | 4 | 20GB | ~5h | 90-93% |

---

## 🔧 高级选项

### 完整消融（7组配置）
```bash
bash scripts/train_a10_full.sh
```

包含：
1. baseline - 纯 Mamba2
2. **mamba2_full** - 推荐生产配置
3. mamba2_sparse - token 稀疏化
4. mamba3_full - 最高精度
5. no_encoder_full - 最快速度
6. hybrid - Mamba2/3 混合
7. attention - Attention 基线

### 自定义训练
```bash
python scripts/train.py \
  --manifest datasets/your_manifest.json \
  --epochs 50 \
  --batch-size 8 \
  --resolution 384 \
  --out runs/custom \
  --temporal-backend mamba \
  --ssm-backend mamba2 \
  --motion-gate \
  --sparse-detect
```

---

## 📖 文档

- **A10_DEPLOYMENT.md** - 完整部署指南
- **MODEL_ARCHITECTURE.md** - 模型架构说明
- **EVALUATION_REPORT.md** - 性能评测报告

---

## ❓ 故障排查

### OOM (显存不足)
```bash
# 降低 batch size
--batch-size 4

# 或降低分辨率
--resolution 256
```

### 训练太慢
检查 GPU 利用率：
```bash
watch -n 1 nvidia-smi
```

应该 >80%，如果很低：
- 确认用了 mamba2 (不是 mamba3)
- 检查 CUDA 版本: `nvcc --version`

### 精度不理想
- 增加 epochs (30 → 50)
- 提高分辨率 (320 → 384)
- 检查数据平衡性

---

## 📞 支持

问题反馈: GitHub Issues
