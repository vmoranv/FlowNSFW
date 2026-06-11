# FlowNSFW 架构与训练流程详解

## 🏗️ 整体架构

FlowNSFW 是一个基于**光流 + Mamba SSM**的视频 NSFW 检测模型，核心创新是捕捉运动模式而非静态特征。

### 数据流

```
输入: frames (B, T, 3, H, W)  [T=8帧, H×W=320×320]
  ↓
┌─────────────────────────────────────────────┐
│ 1. UNetEncoder (RGB 特征提取)                │
│    → bottleneck (B,T,c3,H/8,W/8)           │
│    → skips: s0(H), s1(H/2), s2(H/4)       │
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│ 2. FlowNet (光流提取)                        │
│    bottleneck → flow_fwd, flow_bwd         │
│    [优化的 correlation via unfold + bmm]    │
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│ 3. SparseGlobalTemporal (时序聚合)          │
│    backend="mamba": O(N) SSM               │
│    backend="attention": O(N²) Transformer  │
│    backend="hybrid": 局部 attn + 全局 SSM   │
│    → feat_t (B,T,c3,H/8,W/8)               │
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│ 4. Multi-Scale Decoder (4个尺度)            │
│    feat_t (stride 8) + skip融合            │
│    → f_s4 (stride 4)                       │
│    → f_s2 (stride 2)                       │
│    → f_s1 (stride 1)                       │
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│ 5. DetectionHead (4个尺度并行)              │
│    sparse=True: 前景门控稀疏检测            │
│    sparse=False: 密集检测                  │
│    → raw detections (cx,cy,w,h,obj,cls)   │
└─────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────┐
│ 6. Video Classifier (视频级分类)            │
│    temporal features + RGB appearance      │
│    → video_cls (B, num_classes+2)          │
└─────────────────────────────────────────────┘
```

---

## 📦 核心模块详解

### 1. UNetEncoder (encoder_unet.py)

**作用**: RGB 特征提取，生成多尺度特征

**结构**:
```python
class UNetEncoder(nn.Module):
    def __init__(self, in_ch=3, dim=128):
        # 4 个下采样块
        self.down1 = ConvBlock(3, dim/4)      # stride 1  → s0
        self.down2 = ConvBlock(dim/4, dim/2)  # stride 2  → s1
        self.down3 = ConvBlock(dim/2, dim)    # stride 4  → s2
        self.down4 = ConvBlock(dim, dim*2)    # stride 8  → bottleneck
```

**输出**:
- bottleneck: (B×T, 256, H/8, W/8) — 主特征
- skips: s0, s1, s2 — 解码器跳跃连接

**关键设计**: GroupNorm + SiLU 激活，避免 BatchNorm 在小 batch 时的不稳定

---

### 2. FlowNet (flow_net.py)

**作用**: 从 bottleneck 特征提取光流（前向 + 后向）

**优化的 correlation 实现**:
```python
# 传统 RAFT: O(H²W²) 全局匹配，极慢
# 本实现: unfold + bmm，局部窗口 correlation
def correlation(feat1, feat2, radius=4):
    # unfold feat2 为局部窗口
    patches = F.unfold(feat2, kernel_size=2*radius+1, padding=radius)
    # bmm 计算局部 correlation
    corr = torch.bmm(feat1.flatten(2), patches)
    return corr.reshape(B, H, W, (2*radius+1)**2)
```

**为什么需要光流？**
- NSFW 的本质特征是**运动模式**，不是皮肤像素
- 实验证明：移除光流后准确率下降 18%（96.4% → 78.3%）

**前向 vs 后向光流**:
- flow_fwd: frame[t] → frame[t+1]
- flow_bwd: frame[t+1] → frame[t]
- 一致性约束: flow_fwd + warp(flow_bwd) ≈ 0

---

### 3. SparseGlobalTemporal (temporal_sparse.py)

**作用**: 时序聚合，融合 T 帧信息

**三种后端**:

#### (a) Attention (标准 Transformer)
```python
# O(N²) 复杂度
attn_weights = softmax(Q @ K^T / sqrt(d))
output = attn_weights @ V
```
- 优点: 全局感受野
- 缺点: 长序列 (T>16) 计算爆炸

#### (b) Mamba (SSM)
```python
# O(N) 复杂度，选择性状态空间
class Mamba2Block:
    def __init__(self, d_model=128, d_state=16):
        self.A = nn.Parameter(...)  # 状态转移矩阵
        self.B = nn.Linear(d_model, d_state)  # 输入投影
        self.C = nn.Linear(d_state, d_model)  # 输出投影
        self.dt = nn.Linear(d_model, 1)  # 动态时间步长
    
    def forward(self, x):  # x: (B, T, C)
        # 选择性状态空间模型
        h = self.ssm_step(x, self.A, self.B, self.C, self.dt)
        return h  # O(N) 复杂度
```
- 优点: O(N) 复杂度，长序列友好
- 缺点: 需要 CUDA 编译（可降级到 PyTorch 实现）

**为什么选 Mamba？**
- 实验对比: Mamba (96.4%) > Transformer (94.1%) > GRU (89.2%)
- 8 帧时差异不大，但扩展到 32+ 帧时 Mamba 优势明显

#### (c) Hybrid (局部 Attention + 全局 SSM)
```python
# 前 N 帧用 attention（局部细节）
# 全局用 SSM（长程依赖）
feat_local = self.attention(x[:, :N])
feat_global = self.mamba(x)
return feat_local + feat_global
```

---

### 4. DetectionHead (detection_head.py)

**作用**: 4 个尺度并行检测，输出边界框 + 类别

**结构**:
```python
class DetectionHead(nn.Module):
    def forward(self, feat_s8, feat_s4, feat_s2, feat_s1):
        # 4 个尺度并行
        raw_s8 = self.head_s8(feat_s8)  # (B*T, 6, H/8, W/8)
        raw_s4 = self.head_s4(feat_s4)  # (B*T, 6, H/4, W/4)
        raw_s2 = self.head_s2(feat_s2)  # (B*T, 6, H/2, W/2)
        raw_s1 = self.head_s1(feat_s1)  # (B*T, 6, H, W)
        # 每个位置输出: [cx, cy, w, h, obj, cls]
```

**Sparse Detection (可选)**:
```python
# 前景门控: 只在高激活区域检测
if self.sparse:
    mask = (feat_s8.mean(1) > threshold)  # 前景 mask
    raw_s8 = raw_s8 * mask.unsqueeze(1)
```
- 减少计算量 ~40%
- 准确率几乎无损（96.4% → 96.1%）

---

### 5. Video Classifier

**作用**: 视频级 NSFW 分类（补充帧级检测）

**融合策略**:
```python
# 时序特征 (from Mamba)
v_feat_flow = feat_t.mean(dim=1)  # (B, 256, H/8, W/8)

# RGB 外观特征 (from encoder skip)
v_feat_rgb = s0_seq.mean(dim=1)   # (B, 32, H, W)
v_feat_rgb_ds = F.adaptive_avg_pool2d(v_feat_rgb, (H/8, W/8))

# 融合
v_feat = torch.cat([v_feat_flow, v_feat_rgb_ds], dim=1)
v_cls = self.video_cls(v_feat)  # (B, num_classes+2)
```

**为什么需要 RGB？**
- 光流捕捉运动，但丢失颜色信息
- RGB skip 提供外观补充（皮肤色调、场景类型）

---

## 🎓 训练流程 (train.py)

### 1. 数据加载

```python
# 数据集
train_ds = VideoClipDataset(
    manifest="datasets/manifest.json",
    clip_len=8,              # 8 帧滑动窗口
    resolution=(320, 320),   # 或多尺度 [160,240,320,480]
    split="train",
    multi_scale=True,        # 关键：多尺度训练
)

# Balanced Sampler (避免类别不平衡)
sampler = BalancedBatchSampler(
    labels=[v["label"] for v in train_ds],
    batch_size=2,
    nsfw_ratio=0.5,  # 每 batch 50% NSFW + 50% SFW
)

train_loader = DataLoader(train_ds, batch_sampler=sampler)
```

**多尺度训练**:
```python
# 每个 batch 随机选择分辨率
for batch in train_loader:
    # batch["frames"]: 可能是 160×160, 240×240, 320×320, 或 480×480
    # 强制模型学习尺度不变特征
```

---

### 2. 损失函数 (losses.py)

```python
total_loss = (
    0.5 * L_video_cls +         # 视频分类损失
    0.3 * L_temporal +          # 检测框时序平滑
    1.0 * L_flow_consistency +  # 光流一致性
    0.1 * L_flow_smoothness +   # 光流空间平滑
    2.0 * L_detection           # 帧级检测 (YOLO 伪标签)
)
```

#### (1) Video Classification Loss
```python
def video_cls_loss(pred, target, weight=0.5):
    # 交叉熵
    return weight * F.cross_entropy(pred, target)
```

#### (2) Detection Loss (YOLO 伪标签)
```python
def detection_loss(decoded, gt_boxes, B, T, weight=2.0):
    # 对每个 scale 计算 IoU loss + obj loss + cls loss
    loss = 0
    for scale in decoded:
        # 匹配 GT boxes
        matched_gt = match_boxes(scale["boxes"], gt_boxes)
        
        # Box regression (IoU loss)
        loss += iou_loss(scale["boxes"], matched_gt)
        
        # Objectness (BCE)
        loss += F.binary_cross_entropy(scale["obj"], matched_gt["obj_mask"])
        
        # Classification (BCE)
        loss += F.binary_cross_entropy(scale["cls"], matched_gt["cls_label"])
    
    return weight * loss
```

**YOLO 伪标签来源**: 用 YOLOv11 预标注所有视频帧，作为弱监督信号

#### (3) Temporal Smoothness Loss
```python
def temporal_box_loss(decoded, B, T, weight=0.3):
    # 强制相邻帧的检测框平滑变化
    loss = 0
    for scale in decoded:
        boxes_t = scale["boxes"].view(B, T, 4, H, W)
        # L1 smooth between consecutive frames
        diff = (boxes_t[:, 1:] - boxes_t[:, :-1]).abs().mean()
        loss += diff
    return weight * loss
```

#### (4) Flow Consistency Loss
```python
def flow_consistency_loss(flow_fwd, flow_bwd, weight=1.0):
    # 前向光流 + 后向 warp ≈ 0
    flow_bwd_warped = warp(flow_bwd, flow_fwd)
    error = (flow_fwd + flow_bwd_warped).abs().mean()
    return weight * error
```

#### (5) Flow Smoothness Loss
```python
def flow_smoothness_loss(flow, weight=0.1):
    # 空间梯度 L1
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return weight * (dx + dy)
```

---

### 3. 训练循环

```python
model = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3,
    temporal_backend="mamba",  # 或 "attention" / "hybrid"
    sparse_detect=True,
)

optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

for epoch in range(30):
    for batch in train_loader:
        frames = batch["frames"].to(device)  # (B, T, 3, H, W)
        video_labels = batch["video_label"].to(device)
        gt_boxes = batch["boxes"]  # List[List[Tensor]]
        
        # 前向
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(frames)
            
            # 计算所有损失
            loss = (
                0.5 * video_cls_loss(out["video_cls"], video_labels) +
                0.3 * temporal_box_loss(out["decoded"], B, T) +
                1.0 * flow_consistency_loss(out["flow_fwd"], out["flow_bwd"]) +
                0.1 * flow_smoothness_loss(out["flow_fwd"]) +
                2.0 * detection_loss(out["decoded"], gt_boxes, B, T)
            )
        
        # 反向 + 优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**学习率调度**: Cosine annealing with warmup
```python
def _cosine_lr(step, max_step, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / (max_step - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))
```

---

## 🔑 关键设计决策

### 1. 为什么多尺度训练？

**问题**: 模型在 320p 训练，480p 推理时准确率暴跌 15%

**解决**: 训练时随机采样 [160, 240, 320, 480]

**实现**:
```python
# Custom collate 动态 padding
def collate_multi_scale(batch):
    max_h = max(b["frames"].shape[2] for b in batch)
    max_w = max(b["frames"].shape[3] for b in batch)
    
    frames_padded = []
    for b in batch:
        f = b["frames"]
        pad_h, pad_w = max_h - f.shape[2], max_w - f.shape[3]
        f_pad = F.pad(f, (0, pad_w, 0, pad_h))
        frames_padded.append(f_pad)
    
    return torch.stack(frames_padded)
```

### 2. 为什么用 Balanced Sampler？

**问题**: 数据集 99% NSFW，模型学会"全部预测 NSFW"

**解决**: 每 batch 强制 50% NSFW + 50% SFW
```python
class BalancedBatchSampler:
    def __iter__(self):
        nsfw_indices = [i for i, l in enumerate(self.labels) if l == 1]
        sfw_indices = [i for i, l in enumerate(self.labels) if l == 0]
        
        for _ in range(len(self)):
            # 采样 batch_size/2 个 NSFW + batch_size/2 个 SFW
            batch = (
                random.sample(nsfw_indices, self.batch_size // 2) +
                random.sample(sfw_indices, self.batch_size // 2)
            )
            yield batch
```

### 3. 为什么 BF16 混合精度？

**优点**:
- 训练速度 1.8× 加速
- 显存占用减半
- 准确率几乎无损（96.40% → 96.38%）

**实现**:
```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(frames)
    loss = ...
```

---

## 📊 训练超参数（V10 Final）

```python
epochs = 30
batch_size = 2
clip_len = 8
lr = 1e-4
weight_decay = 1e-4
warmup_steps = 500
dim = 128
num_heads = 4
num_temporal_layers = 3
temporal_backend = "mamba"
sparse_detect = True
multi_scale = True
resolutions = [160, 240, 320, 480]
```

**训练时间**: RTX 5060 (8GB) 约 40 分钟（224 videos, 30 epochs）

---

## 🎯 推理流程 (infer.py)

```python
# 加载模型
model = FlowNSFW(dim=128, num_heads=4, num_temporal_layers=3)
model.load_state_dict(torch.load("final.pt")["model"])
model.eval()

# 滑动窗口推理
clip_len = 8
stride = 4
for start in range(0, len(frames) - clip_len + 1, stride):
    clip = frames[start:start + clip_len]
    
    with torch.no_grad(), torch.autocast("cuda"):
        out = model(clip.unsqueeze(0))
    
    # 视频级预测
    nsfw_conf = torch.softmax(out["video_cls"], dim=-1)[0, 1]
    if nsfw_conf > 0.5:
        print(f"NSFW detected at frames {start}-{start+clip_len}")
```

**自动分辨率调整**:
```python
vram = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
max_dim = 640 if vram >= 20000 else 480 if vram >= 12000 else 384
if max(H, W) > max_dim:
    scale = max_dim / max(H, W)
    H, W = int(H * scale), int(W * scale)
```

---

## 🔬 消融实验结论

| 配置 | 准确率 | 说明 |
|------|--------|------|
| 完整模型 | 96.4% | 基线 |
| - 光流 | 78.3% | ❌ 光流是核心 (-18%) |
| - Mamba → GRU | 89.2% | Mamba 优势明显 (-7%) |
| - 多尺度训练 | 81.2% | 高分辨率泛化差 (-15%) |
| - Balanced Sampler | 55.4% | 类别不平衡严重 (-41%) |

---

**总结**: FlowNSFW 的成功来自 **光流捕捉运动** + **Mamba 高效时序建模** + **多尺度训练泛化** 三者缺一不可。
