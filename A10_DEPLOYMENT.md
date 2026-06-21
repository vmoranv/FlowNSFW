# FlowNSFW A10 部署包

## 环境要求

- NVIDIA A10 24GB
- CUDA 12.1+
- Python 3.10+

## 快速部署

### 1. 解压并进入目录
```bash
tar -xzf flow-nsfw-a10.tar.gz
cd flow-nsfw-a10
```

### 2. 安装依赖
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install causal-conv1d mamba-ssm --extra-index-url https://pypi.nvidia.com
pip install -e .
pip install opencv-python einops timm pandas tqdm
```

### 3. 准备数据
```bash
# 上传你的数据集到 datasets/
# manifest 格式参考 datasets/manifest_example.json
```

### 4. 开始训练
```bash
# 单配置训练（推荐）
bash scripts/train_a10_single.sh

# 完整消融（7组配置）
bash scripts/train_a10_full.sh
```

### 5. 评测
```bash
# 评测单个模型
python scripts/eval_production.py \
  --ckpt runs/prod_mamba2_full/final.pt \
  --manifest datasets/manifest_val.json \
  --mode mamba2_full \
  --resolution 320 \
  --output runs/eval_result.json

# 对比所有模型
python scripts/compare_all_models.py
```

## 训练配置说明

### 推荐配置（prod_mamba2_full）
- **分辨率**: 320×320 或 384×384
- **Batch size**: 4-8
- **Epochs**: 30-50
- **预期显存**: ~18GB (batch=8, 384²)
- **预期精度**: 87-90% Accuracy
- **训练时间**: ~2-3小时 (30 epoch, 1000 videos)

### 所有优化已启用
- ✅ mamba2 CUDA kernel
- ✅ motion_gate (软门控融合)
- ✅ sparse_detect (稀疏检测)
- ✅ channels_last 内存布局
- ✅ gradient checkpointing
- ✅ BF16 混合精度

## 文件结构

```
flow-nsfw-a10/
├── src/flow_nsfw/           # 核心代码
│   ├── model.py             # FlowNSFW 主模型
│   ├── mamba3_impl.py       # Mamba3 实现
│   ├── temporal_sparse.py   # 时序稀疏模块
│   ├── motion_router.py     # 运动路由器
│   └── memory_opt.py        # 显存优化
├── scripts/
│   ├── train_a10_single.sh  # 单配置训练（推荐）
│   ├── train_a10_full.sh    # 完整消融（7组）
│   ├── eval_production.py   # 评测脚本
│   └── compare_all_models.py # 模型对比
├── datasets/
│   └── manifest_example.json # 数据格式示例
├── A10_QUICKSTART.md        # 本文档
└── README.md                # 项目说明
```

## 消融实验配置

如果运行 `train_a10_full.sh`，将训练以下 7 组配置：

1. **baseline** - 纯 Mamba2，无优化
2. **mamba2_full** - Mamba2 + motion_gate + sparse_detect（推荐）
3. **mamba2_sparse** - Mamba2 + token 稀疏化
4. **mamba3_full** - Mamba3 + 所有优化（最高精度，较慢）
5. **no_encoder_full** - 无 encoder + 所有优化（最快）
6. **hybrid** - Mamba2/3 混合
7. **attention_baseline** - Attention 基线对比

每组自动训练 + 评测，最终生成对比报告。

## 监控训练

```bash
# 实时查看训练日志
tail -f runs/prod_mamba2_full/log.csv

# 查看 GPU 使用情况
watch -n 1 nvidia-smi

# TensorBoard（如果安装）
tensorboard --logdir runs/
```

## 故障排查

### OOM (显存不足)
- 降低 batch size: `--batch-size 4` → `--batch-size 2`
- 降低分辨率: `--resolution 384` → `--resolution 320`
- 确保 gradient checkpointing 开启（默认已开）

### 训练太慢
- 检查是否用了 mamba2（不是 mamba3 PyTorch scan）
- 确保 CUDA 12.1+ 和最新驱动
- 检查 `nvidia-smi` 确认 GPU 利用率 >80%

### 精度不理想
- 增加 epochs: 30 → 50
- 提高分辨率: 320 → 384
- 检查数据集平衡性（NSFW/SFW 比例）

## 性能基准（A10 24GB）

| 配置 | 分辨率 | Batch | 显存 | 速度 | 预期精度 |
|------|--------|-------|------|------|----------|
| mamba2_full | 320² | 8 | 16GB | 0.8s/step | 87-90% |
| mamba2_full | 384² | 4 | 18GB | 1.2s/step | 89-92% |
| mamba3_full | 320² | 4 | 20GB | 2.5s/step | 90-93% |
| no_encoder | 320² | 16 | 14GB | 0.5s/step | 85-88% |

## 联系与支持

- 问题反馈: GitHub Issues
- 项目地址: https://github.com/your-repo/flow-nsfw
