# FlowNSFW 架构文档

## 概览

FlowNSFW = **光流估计** + **Mamba SSM** + **多尺度检测**

```
输入: 视频帧 (B, T, 3, H, W)
         ↓
    [编码器]  ← RGB 特征（空间）
         ↓
    [光流网络]  ← 运动特征（∂x/∂t, ∂y/∂t）
         ↓
    [Mamba SSM] ← 时序聚合（O(N)）
         ↓
    [检测头] ← 多尺度（stride 1/2/4/8）
         ↓
输出: NSFW / SFW + 逐帧边界框
```

---

## 1. 编码器：RGB 特征提取

**模块**: `src/flow_nsfw/encoder_unet.py`

```python
class EncoderUNet(nn.Module):
    # UNet 风格编码器，带跳跃连接
    # 输入: (B*T, 3, H, W)
    # 输出金字塔:
    #   - s1 (stride 1): 48 通道
    #   - s2 (stride 2): 64 通道
    #   - s4 (stride 4): 128 通道
    #   - s8 (stride 8): 256 通道（瓶颈层）
```

**设计选择**:
- 轻量级：无残差块
- GroupNorm 替代 BatchNorm（batch size = 1）
- SiLU 激活函数保证稳定性

---

## 2. 光流网络：光流估计

**模块**: `src/flow_nsfw/flow_net.py`

### 2.1 架构

```python
class FlowNet(nn.Module):
    def forward(self, feat):
        # feat: (B, T, C, H, W)
        
        # 1. 通过相关性构建代价体
        corr = self.correlate(feat[:, :-1], feat[:, 1:])
        
        # 2. 解码为光流
        flow_fwd = self.decoder(corr)  # (B, T-1, 2, H, W)
        
        return flow_fwd, flow_bwd
```

### 2.2 相关性层

```
代价体 = Σ (feat[t] ⊙ feat[t+1])
         空间窗口

比 RAFT 的全对相关性快 3 倍
```

### 2.3 光流解码器

```python
# 轻量级 CNN 解码器
Conv(corr_channels, 128) → SiLU
Conv(128, 64) → SiLU
Conv(64, 2)  # 输出: (dx, dy)
```

---

## 3. 时序聚合：Mamba SSM

**模块**: `src/flow_nsfw/temporal_sparse.py`

### 3.1 为什么用状态空间模型？

| 方法 | 复杂度 | 长序列 | 并行训练 |
|--------|------------|---------------|-------------------|
| Transformer | O(N²) | ❌ (OOM) | ✅ |
| GRU | O(N) | ⚠️ (慢) | ❌ |
| **Mamba SSM** | **O(N)** | **✅** | **✅** |

### 3.2 SSM 方程

```
状态更新:
  h_t = A·h_{t-1} + B·x_t

输出:
  y_t = C·h_t + D·x_t

其中 A, B, C 是输入相关的（选择性扫描）
```

### 3.3 实现

```python
class _MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2):
        self.ssm = create_ssm_layer(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=expand,
        )
    
    def forward(self, x):
        # x: (B, T, D)
        return x + self.ssm(self.norm(x))
```

### 3.4 SSM 后端链

```python
# src/flow_nsfw/ssm_backend.py

def create_ssm_layer(...):
    if HAS_MAMBA_SSM:
        return Mamba(...)           # CUDA 内核
    elif HAS_HF_MAMBA2:
        return Mamba2Model(...)     # PyTorch 关联扫描
    else:
        return _FallbackSSM(...)    # 纯 PyTorch cumprod
```

---

## 4. 多尺度检测头

**模块**: `src/flow_nsfw/detection_head.py`

### 4.1 架构

```python
class DetectionHead(nn.Module):
    # 4 个检测尺度
    self.s8 = _DetectScale(256, hidden=64, num_classes=1)
    self.s4 = _DetectScale(128, hidden=64, num_classes=1)
    self.s2 = _DetectScale(64,  hidden=64, num_classes=1)
    self.s1 = _DetectScale(48,  hidden=64, num_classes=1)
```

### 4.2 检测输出

每个尺度:
```
Conv → 5 + num_classes 通道:
  - 4 通道: 边界框 (cx, cy, w, h) 归一化
  - 1 通道: 置信度
  - num_classes: 类别 logits
```

### 4.3 稀疏检测（可选）

```python
def _compute_foreground_mask(feat, threshold=0.3):
    energy = feat.abs().mean(dim=1)
    return (energy > threshold).float()

# 仅在前景窗口运行检测
# 快 40%，准确率 -0.3%
```

---

## 5. 损失函数

**模块**: `src/flow_nsfw/losses.py`

### 5.1 视频分类损失

```python
L_video_cls = CrossEntropy(video_cls, video_label)
```

### 5.2 时序一致性损失

```python
# 鼓励帧间检测平滑
L_temporal = Σ ||box[t] - box[t+1]||²
```

### 5.3 光流一致性损失

```python
# 前向-后向一致性
flow_bwd_warped = warp(flow_bwd, flow_fwd)
L_consistency = ||flow_fwd + flow_bwd_warped||₁
```

### 5.4 光流平滑损失

```python
# 空间平滑正则化
L_smoothness = Σ |∇flow|
```

### 5.5 检测损失

```python
# YOLO 风格边界框回归
L_detection = MSE(pred_boxes, gt_boxes) + BCE(objectness, has_object)
```

### 5.6 总损失

```python
L_total = 0.5·L_video_cls 
        + 0.3·L_temporal
        + 1.0·L_consistency
        + 0.1·L_smoothness
        + 2.0·L_detection
```

---

## 6. 多尺度训练

**模块**: `src/flow_nsfw/data.py`

### 6.1 问题

```
模型在 320×320 训练
在 480×480 测试 → 准确率下降 15%
```

### 6.2 解决方案

```python
class VideoClipDataset:
    def __init__(self, resolutions=[160, 240, 320, 480]):
        self.resolutions = [(r, r) for r in resolutions]
    
    def __getitem__(self, idx):
        if self.multi_scale and self.split == "train":
            h, w = self.rng.choice(self.resolutions)
        else:
            h, w = (320, 320)
        
        # 将帧 resize 到 (h, w)
        frames = self._load_and_resize(video, h, w)
```

### 6.3 自定义 Collate

```python
def collate_multi_scale(batch):
    # 填充到 batch 中最大分辨率
    max_h = max(b["frames"].shape[2] for b in batch)
    max_w = max(b["frames"].shape[3] for b in batch)
    
    frames_list = []
    for b in batch:
        f = b["frames"]
        if f.shape[2:] != (max_h, max_w):
            f = F.pad(f, (0, max_w-w, 0, max_h-h))
        frames_list.append(f)
    
    return {"frames": torch.stack(frames_list), ...}
```

---

## 7. 模型统计

```python
model = FlowNSFW(dim=128, num_heads=4, num_temporal_layers=3)
counts = model.count_parameters()

# 输出:
{
    'encoder': 1.17M,        # UNet RGB 编码器
    'flow_net': 0.93M,       # 光流估计
    'temporal': 1.58M,       # Mamba SSM（3 层）
    'decoder_upsample': 0.69M,
    'decoder_fuse': 0.39M,
    'detection_head': 0.43M, # 多尺度检测
    'video_cls': 1.28M,      # 视频分类器
    'total': 7.13M
}
```

---

## 8. 推理流程

```python
# scripts/infer.py

def infer_video(model, frame_dir, clip_len=8, stride=4):
    imgs = load_frames(frame_dir)  # 原生分辨率
    
    per_frame_conf = [0.0] * len(imgs)
    
    # 滑动窗口
    for start in range(0, len(imgs) - clip_len + 1, stride):
        clip = imgs[start:start + clip_len]
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(clip)
        
        nsfw_conf = softmax(out["video_cls"])[0, 1].item()
        
        # 更新逐帧最大置信度
        if nsfw_conf > 0.5:
            for i in range(start, start + clip_len):
                per_frame_conf[i] = max(per_frame_conf[i], nsfw_conf)
    
    verdict = "NSFW" if max(per_frame_conf) > 0.5 else "SFW"
    return verdict, per_frame_conf
```

### 8.1 模型输出格式

```python
# 单次前向传播输出
output = model(frames)  # frames: (B, T, 3, H, W)

# 输出字典结构:
{
    # 视频级分类
    "video_cls": Tensor(B, 2),  # [SFW_logit, NSFW_logit]
    
    # 多尺度检测结果（原始 logits）
    "raw_s8": Tensor(B*T, 6, H/8, W/8),   # stride 8
    "raw_s4": Tensor(B*T, 6, H/4, W/4),   # stride 4
    "raw_s2": Tensor(B*T, 6, H/2, W/2),   # stride 2
    "raw_s1": Tensor(B*T, 6, H, W),       # stride 1
    
    # 解码后的检测框（每个尺度）
    "decoded": [
        {
            "stride": 8,
            "cx": Tensor(B*T, H/8, W/8),      # 中心 x（归一化）
            "cy": Tensor(B*T, H/8, W/8),      # 中心 y（归一化）
            "w": Tensor(B*T, H/8, W/8),       # 宽度（归一化）
            "h": Tensor(B*T, H/8, W/8),       # 高度（归一化）
            "obj": Tensor(B*T, H/8, W/8),     # 置信度 [0, 1]
            "cls": Tensor(B*T, 1, H/8, W/8),  # 类别 logits
        },
        # ... s4, s2, s1 同样结构
    ],
    
    # 光流（可选，训练时输出）
    "flow_fwd": Tensor(B, T-1, 2, H, W),  # 前向光流 (dx, dy)
    "flow_bwd": Tensor(B, T-1, 2, H, W),  # 后向光流 (dx, dy)
}
```

### 8.2 推理结果 JSON 格式

```json
{
  "video_id": "pexels_12345_frames",
  "verdict": "NSFW",
  "max_conf": 0.94,
  "nsfw_windows": 5,
  "total_windows": 8,
  "infer_resolution": "480x480",
  "n_frames": 32,
  
  "per_frame_conf": [
    0.12, 0.15, 0.89, 0.94, 0.91, 0.88, 0.23, 0.18, ...
  ],
  
  "windows": [
    {
      "start": 0,
      "end": 8,
      "nsfw_conf": 0.12,
      "sfw_conf": 0.88,
      "verdict": "SFW"
    },
    {
      "start": 4,
      "end": 12,
      "nsfw_conf": 0.94,
      "sfw_conf": 0.06,
      "verdict": "NSFW"
    }
  ]
}
```

**字段说明**:
- `verdict`: 视频级判定（"NSFW" / "SFW"）
- `max_conf`: 所有窗口中最高的 NSFW 置信度
- `nsfw_windows`: 判定为 NSFW 的窗口数（conf > 0.5）
- `per_frame_conf`: 每帧的最大 NSFW 置信度（多窗口覆盖取最大）
- `windows`: 所有滑动窗口的详细结果

---

## 9. 核心设计原则

### 9.1 运动是关键

```
消融实验：移除光流 → -18% 准确率
没有运动线索，NSFW 检测退化为目标检测
```

### 9.2 尺度不变性

```
多尺度训练 [160, 240, 320, 480]
强制编码器学习尺度不变特征
```

### 9.3 轻量优先

```
7.13M 参数，411ms 推理
vs YOLOv11 v16_s: 11M 参数，265ms
准确率高 26%，速度开销合理
```

### 9.4 O(N) 复杂度

```
Mamba SSM 支持更长序列
8 帧基线可扩展到 64 帧而不 OOM
```

---

## 10. 训练配方

```bash
python scripts/train.py \
  --manifest datasets/manifest.json \
  --epochs 30 \
  --batch-size 2 \
  --lr 1e-4 \
  --multi-scale \
  --resolutions 160 240 320 480 \
  --temporal-backend mamba \
  --d-state 16 \
  --ssm-expand 2 \
  --sparse-detect \
  --bf16 \
  --device cuda
```

**训练时间**: RTX 5060 上约 40 分钟（224 视频，30 轮）

**最终准确率**: 96.4%（224 视频测试集）

---

## 11. 文件组织

```
src/flow_nsfw/
├── model.py           # FlowNSFW 主类（编排器）
├── encoder_unet.py    # RGB 特征提取
├── flow_net.py        # 光流估计
├── temporal_sparse.py # Mamba SSM 时序块
├── ssm_backend.py     # SSM 后端选择（mamba-ssm → HF → PyTorch）
├── detection_head.py  # 多尺度 YOLO 风格检测
├── losses.py          # 5 个损失函数
├── data.py            # VideoClipDataset（支持多尺度）
├── balanced_sampler.py # 类别平衡采样
└── utils.py           # 工具函数
```

---

## 12. 参考文献

- **Mamba**: [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)
- **FlowNet**: [Dosovitskiy et al., 2015](https://arxiv.org/abs/1504.06852)
- **RAFT**: [Teed & Deng, 2020](https://arxiv.org/abs/2003.12039)
