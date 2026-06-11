# FlowNSFW 框架整理总结

## 🎯 当前状态

### V2 模型性能（最佳）
- **参数量**: 5.22M
- **训练**: 4040 steps, 40 epochs
- **验证集表现** (16 samples):
  - 240×240: ✅ **100% 准确率**（6/6 NSFW, 10/10 SFW）
  - 320×320: ✅ **100% 准确率**
  - 480×480: ⚠️ **81.2% 准确率**（3个SFW误报）
  - 640×640: ⚠️ **81.2% 准确率**（3个SFW误报）

- **推理速度**: 35.4 ms/clip @ 240×240 (RTX 5060)
- **模型路径**: `D:/cumhub/flow-nsfw/runs/flow_nsfw_v2/final.pt`

### 核心问题诊断

1. **高分辨率误报** ⚠️
   - 原因：训练固定 160×160 分辨率，未见过高分辨率
   - 影响：480p+ 出现 3/10 SFW 误报

2. **SFW 数据质量差** 🎯
   - 100 个 SFW 都是单帧重复（无真实运动）
   - 模型没学到 SFW 视频的真实运动模式

3. **光流特征未激活** 🌊
   - `L_temporal ≈ 0` 全程训练
   - FlowNet 形同虚设，运动信息未利用

---

## ✅ 已完成的改进

### 1. 多尺度训练支持 ✅
**文件**: `src/flow_nsfw/data.py`, `scripts/train.py`

**改进内容**:
```python
# 支持随机多分辨率训练
VideoClipDataset(
    resolution=[(160,160), (240,240), (320,320), (480,480)],
    multi_scale=True,
)
```

**使用方法**:
```bash
python scripts/train.py \
  --manifest datasets/manifest.json \
  --multi-scale \
  --resolutions 160 240 320 480 \
  ...
```

**预期提升**: 480×480 准确率 81% → **95%+**

---

### 2. 真实 SFW 视频采集工具 ✅
**文件**: `scripts/collect_sfw_videos.py`, `scripts/merge_manifests.py`

**功能**:
- Pexels API 自动下载（免费，CC0）
- 自动帧提取（ffmpeg）
- Manifest 合并工具

**使用流程**:
```bash
# Step 1: 获取 Pexels API key (https://www.pexels.com/api/)

# Step 2: 采集 500 个 SFW 视频
python scripts/collect_sfw_videos.py \
  --source pexels \
  --count 500 \
  --out datasets/sfw_videos/ \
  --pexels-key YOUR_API_KEY \
  --frames-per-video 60 \
  --fps 2

# Step 3: 合并到训练 manifest
python scripts/merge_manifests.py \
  --base datasets/manifest_final.json \
  --new datasets/sfw_videos/sfw_manifest.json \
  --out datasets/manifest_v3_multiscale.json \
  --label 0
```

**预期收益**:
- 解决高分辨率误报根本原因
- SFW 数据从 100 → **600+**
- 真实运动模式覆盖

---

### 3. 光流一致性损失 ✅
**文件**: `src/flow_nsfw/losses.py`

**新增损失**:
```python
# Forward-backward 一致性
L_flow_consistency = ||flow_fwd + warp(flow_bwd)||

# 空间平滑性
L_flow_smoothness = ||∇x flow|| + ||∇y flow||
```

**权重配置**:
```python
LossWeights(
    flow_consistency=1.0,  # 前后向一致性
    flow_smoothness=0.1,   # 空间平滑
)
```

**预期提升**: 利用运动信息，提升召回率 +5%

---

### 4. Detection Head 监督 ✅
**文件**: `src/flow_nsfw/data.py`, `src/flow_nsfw/losses.py`

**改进内容**:
- 数据加载器返回 YOLO pseudo-label boxes
- 简化的 detection loss（基于 MSE）
- Per-frame box 级别监督

**损失函数**:
```python
simple_detection_loss(
    decoded,      # 模型预测的 boxes
    gt_boxes,     # YOLO pseudo-labels
    weight=2.0,
)
```

**预期提升**: Per-frame 精细化监督，提升精度 +3%

---

## 🚀 完整训练命令（V3 多尺度 + 光流 + Detection）

```bash
# 使用新的完整框架训练
D:/cumhub/anti-nsfw-yolo/.venv2/Scripts/python.exe scripts/train.py \
  --manifest datasets/manifest_v3_multiscale.json \
  --epochs 40 \
  --batch-size 2 \
  --clip-len 4 \
  --lr 2e-4 \
  --dim 128 \
  --num-heads 4 \
  --num-temporal-layers 3 \
  --topk-global 64 \
  --multi-scale \
  --resolutions 160 240 320 480 \
  --log-every 20 \
  --ckpt-every 3000 \
  --out runs/flow_nsfw_v3_final \
  --bf16 \
  --device cuda \
  --seed 42
```

---

## 📊 预期性能提升

| 指标 | V2 (当前) | V3 (预期) | 提升 |
|------|----------|----------|------|
| **240×240 准确率** | 100% | 100% | - |
| **480×480 准确率** | 81.2% | **95%+** | +14% |
| **480×480 误报** | 3/10 | **0/10** | -100% |
| **召回率** | 100% | 100% | - |
| **光流激活** | ❌ | ✅ | 新增 |
| **Detection 监督** | ❌ | ✅ | 新增 |
| **数据集规模** | 224 videos | **724+ videos** | +3.2× |
| **真实 SFW** | 0 | **500+** | 从无到有 |

---

## 📝 改进清单

### ✅ 已完成 (P0 优先级)
- [x] 多尺度训练支持
- [x] SFW 视频采集工具
- [x] Manifest 合并脚本
- [x] 光流一致性损失
- [x] 光流平滑损失
- [x] Detection Head 监督
- [x] 数据加载器返回 GT boxes

### 🔄 下一步 (执行顺序)
1. **采集 SFW 数据** (2-4 小时)
   ```bash
   python scripts/collect_sfw_videos.py --pexels-key YOUR_KEY --count 500
   ```

2. **合并 Manifest** (1 分钟)
   ```bash
   python scripts/merge_manifests.py --base ... --new ... --out ...
   ```

3. **V3 训练** (30 分钟)
   ```bash
   python scripts/train.py --manifest datasets/manifest_v3_multiscale.json --multi-scale ...
   ```

4. **评估 V3** (5 分钟)
   ```bash
   python scripts/eval_v3.py --ckpt runs/flow_nsfw_v3_final/final.pt
   ```

---

## 🎓 技术细节

### 多尺度训练原理
- 训练时每个 batch 随机选择分辨率
- 强制模型学习跨尺度的不变特征
- 解决高分辨率泛化问题

### 光流一致性
- Forward flow: frame[t] → frame[t+1]
- Backward flow: frame[t+1] → frame[t]
- 一致性约束: `flow_fwd + warp(flow_bwd) ≈ 0`

### Detection 监督
- YOLO 伪标签提供 per-frame box GT
- 模型预测的 detection head 与 GT 做 MSE loss
- 比纯分类监督更细粒度

---

## 📦 产物清单

### 代码文件
- `src/flow_nsfw/data.py` — 多尺度支持 + GT boxes
- `src/flow_nsfw/losses.py` — 光流损失 + detection loss
- `scripts/train.py` — 完整训练流程
- `scripts/collect_sfw_videos.py` — SFW 采集工具
- `scripts/merge_manifests.py` — Manifest 合并
- `docs/SFW_COLLECTION_GUIDE.md` — 使用指南

### 模型检查点
- `runs/flow_nsfw_v2/final.pt` — 当前最佳 (100% @ 240p)
- `runs/flow_nsfw_v3_final/` — 待训练（多尺度 + 光流）

---

## 🔬 实验建议

### A/B 测试
1. **V2 (baseline)**: 当前模型
2. **V3-multi**: V2 + 多尺度训练
3. **V3-flow**: V3-multi + 光流损失
4. **V3-full**: V3-flow + detection 监督 + 500 SFW

### 评估指标
- 240×240, 320×320, 480×480, 640×640 四个分辨率
- Accuracy, Precision, Recall, F1
- 误报案例分析（哪些 SFW 被误判）

---

## 🎯 最终目标

**生产可用标准**:
- ✅ 240×240: 100% 准确率
- ✅ 320×320: 100% 准确率
- 🎯 480×480: **100% 准确率**（当前 81%）
- 🎯 640×640: **95%+ 准确率**（当前 81%）
- ✅ 推理速度: < 50ms/clip
- ✅ 0 误报 @ 240-320p

**达成路径**: 执行上述"下一步"1-4

---

## 📞 快速启动

```bash
# 1. 立即可做：采集 SFW（需要 Pexels key）
python scripts/collect_sfw_videos.py --pexels-key YOUR_KEY --count 500 --out datasets/sfw_videos/

# 2. 合并数据
python scripts/merge_manifests.py --base datasets/manifest_final.json --new datasets/sfw_videos/sfw_manifest.json --out datasets/manifest_v3.json --label 0

# 3. 训练 V3
python scripts/train.py --manifest datasets/manifest_v3.json --multi-scale --out runs/flow_nsfw_v3_final

# 4. 评估
python -c "
import torch
from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset
# ... 评估代码 ...
"
```

**预计总耗时**: 3-4 小时（采集 2h + 训练 30min + 评估 5min）
