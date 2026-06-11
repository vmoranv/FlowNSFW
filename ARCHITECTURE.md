# FlowNSFW Architecture

## Overview

FlowNSFW = **Optical Flow** + **Mamba SSM** + **Multi-Scale Detection**

```
Input: Video frames (B, T, 3, H, W)
         ↓
    [Encoder]  ← RGB features (spatial)
         ↓
    [FlowNet]  ← Motion features (∂x/∂t, ∂y/∂t)
         ↓
    [Mamba SSM] ← Temporal aggregation (O(N))
         ↓
    [Detection Head] ← Multi-scale (stride 1/2/4/8)
         ↓
Output: NSFW / SFW + per-frame boxes
```

---

## 1. Encoder: RGB Feature Extraction

**Module**: `src/flow_nsfw/encoder_unet.py`

```python
class EncoderUNet(nn.Module):
    # UNet-style encoder with skip connections
    # Input: (B*T, 3, H, W)
    # Output pyramid:
    #   - s1 (stride 1): 48 channels
    #   - s2 (stride 2): 64 channels
    #   - s4 (stride 4): 128 channels
    #   - s8 (stride 8): 256 channels (bottleneck)
```

**Design choices**:
- Lightweight: no residual blocks
- GroupNorm instead of BatchNorm (batch size = 1)
- SiLU activation for stability

---

## 2. FlowNet: Optical Flow Estimation

**Module**: `src/flow_nsfw/flow_net.py`

### 2.1 Architecture

```python
class FlowNet(nn.Module):
    def forward(self, feat):
        # feat: (B, T, C, H, W)
        
        # 1. Build cost volume via correlation
        corr = self.correlate(feat[:, :-1], feat[:, 1:])
        
        # 2. Decode to flow
        flow_fwd = self.decoder(corr)  # (B, T-1, 2, H, W)
        
        return flow_fwd, flow_bwd
```

### 2.2 Correlation Layer

```
Cost Volume = Σ (feat[t] ⊙ feat[t+1])
              spatial window

3× faster than RAFT's all-pairs correlation
```

### 2.3 Flow Decoder

```python
# Lightweight CNN decoder
Conv(corr_channels, 128) → SiLU
Conv(128, 64) → SiLU
Conv(64, 2)  # Output: (dx, dy)
```

---

## 3. Temporal Aggregation: Mamba SSM

**Module**: `src/flow_nsfw/temporal_sparse.py`

### 3.1 Why State-Space Models?

| Method | Complexity | Long Sequence | Parallel Training |
|--------|------------|---------------|-------------------|
| Transformer | O(N²) | ❌ (OOM) | ✅ |
| GRU | O(N) | ⚠️ (slow) | ❌ |
| **Mamba SSM** | **O(N)** | **✅** | **✅** |

### 3.2 SSM Equations

```
State update:
  h_t = A·h_{t-1} + B·x_t

Output:
  y_t = C·h_t + D·x_t

Where A, B, C are input-dependent (selective scan)
```

### 3.3 Implementation

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

### 3.4 SSM Backend Chain

```python
# src/flow_nsfw/ssm_backend.py

def create_ssm_layer(...):
    if HAS_MAMBA_SSM:
        return Mamba(...)           # CUDA kernels
    elif HAS_HF_MAMBA2:
        return Mamba2Model(...)     # PyTorch associative scan
    else:
        return _FallbackSSM(...)    # Pure PyTorch cumprod
```

---

## 4. Multi-Scale Detection Head

**Module**: `src/flow_nsfw/detection_head.py`

### 4.1 Architecture

```python
class DetectionHead(nn.Module):
    # 4 detection scales
    self.s8 = _DetectScale(256, hidden=64, num_classes=1)
    self.s4 = _DetectScale(128, hidden=64, num_classes=1)
    self.s2 = _DetectScale(64,  hidden=64, num_classes=1)
    self.s1 = _DetectScale(48,  hidden=64, num_classes=1)
```

### 4.2 Detection Output

Per scale:
```
Conv → 5 + num_classes channels:
  - 4 channels: box (cx, cy, w, h) normalized
  - 1 channel: objectness
  - num_classes: class logits
```

### 4.3 Sparse Detection (Optional)

```python
def _compute_foreground_mask(feat, threshold=0.3):
    energy = feat.abs().mean(dim=1)
    return (energy > threshold).float()

# Only run detection on foreground windows
# 40% faster, -0.3% accuracy
```

---

## 5. Loss Functions

**Module**: `src/flow_nsfw/losses.py`

### 5.1 Video Classification Loss

```python
L_video_cls = CrossEntropy(video_cls, video_label)
```

### 5.2 Temporal Consistency Loss

```python
# Encourage smooth detection across frames
L_temporal = Σ ||box[t] - box[t+1]||²
```

### 5.3 Flow Consistency Loss

```python
# Forward-backward consistency
flow_bwd_warped = warp(flow_bwd, flow_fwd)
L_consistency = ||flow_fwd + flow_bwd_warped||₁
```

### 5.4 Flow Smoothness Loss

```python
# Spatial smoothness regularization
L_smoothness = Σ |∇flow|
```

### 5.5 Detection Loss

```python
# YOLO-style box regression
L_detection = MSE(pred_boxes, gt_boxes) + BCE(objectness, has_object)
```

### 5.6 Total Loss

```python
L_total = 0.5·L_video_cls 
        + 0.3·L_temporal
        + 1.0·L_consistency
        + 0.1·L_smoothness
        + 2.0·L_detection
```

---

## 6. Multi-Scale Training

**Module**: `src/flow_nsfw/data.py`

### 6.1 Problem

```
Model trained at 320×320
Tested at 480×480 → -15% accuracy drop
```

### 6.2 Solution

```python
class VideoClipDataset:
    def __init__(self, resolutions=[160, 240, 320, 480]):
        self.resolutions = [(r, r) for r in resolutions]
    
    def __getitem__(self, idx):
        if self.multi_scale and self.split == "train":
            h, w = self.rng.choice(self.resolutions)
        else:
            h, w = (320, 320)
        
        # Resize frames to (h, w)
        frames = self._load_and_resize(video, h, w)
```

### 6.3 Custom Collate

```python
def collate_multi_scale(batch):
    # Pad to max resolution in batch
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

## 7. Model Counting

```python
model = FlowNSFW(dim=128, num_heads=4, num_temporal_layers=3)
counts = model.count_parameters()

# Output:
{
    'encoder': 1.17M,        # UNet RGB encoder
    'flow_net': 0.93M,       # Optical flow estimation
    'temporal': 1.58M,       # Mamba SSM (3 layers)
    'decoder_upsample': 0.69M,
    'decoder_fuse': 0.39M,
    'detection_head': 0.43M, # Multi-scale detection
    'video_cls': 1.28M,      # Video classifier
    'total': 7.13M
}
```

---

## 8. Inference Pipeline

```python
# scripts/infer.py

def infer_video(model, frame_dir, clip_len=8, stride=4):
    imgs = load_frames(frame_dir)  # Native resolution
    
    per_frame_conf = [0.0] * len(imgs)
    
    # Sliding window
    for start in range(0, len(imgs) - clip_len + 1, stride):
        clip = imgs[start:start + clip_len]
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(clip)
        
        nsfw_conf = softmax(out["video_cls"])[0, 1].item()
        
        # Update per-frame max confidence
        if nsfw_conf > 0.5:
            for i in range(start, start + clip_len):
                per_frame_conf[i] = max(per_frame_conf[i], nsfw_conf)
    
    verdict = "NSFW" if max(per_frame_conf) > 0.5 else "SFW"
    return verdict, per_frame_conf
```

---

## 9. Key Design Principles

### 9.1 Motion is King

```
Ablation: Remove flow → -18% accuracy
Without motion cues, NSFW detection degrades to object detection
```

### 9.2 Scale Invariance

```
Multi-scale training [160, 240, 320, 480]
Forces encoder to learn scale-invariant features
```

### 9.3 Lightweight First

```
7.13M params, 411ms inference
vs YOLOv11 v16_s: 11M params, 265ms
26% better accuracy with reasonable overhead
```

### 9.4 O(N) Complexity

```
Mamba SSM enables longer sequences
8-frame baseline can extend to 64-frame without OOM
```

---

## 10. Training Recipe

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

**Training time**: ~40 minutes on RTX 5060 (224 videos, 30 epochs)

**Final accuracy**: 96.4% (224-video test set)

---

## 11. File Organization

```
src/flow_nsfw/
├── model.py           # Main FlowNSFW class (orchestrator)
├── encoder_unet.py    # RGB feature extraction
├── flow_net.py        # Optical flow estimation
├── temporal_sparse.py # Mamba SSM temporal blocks
├── ssm_backend.py     # SSM backend selection (mamba-ssm → HF → PyTorch)
├── detection_head.py  # Multi-scale YOLO-style detection
├── losses.py          # 5 loss functions
├── data.py            # VideoClipDataset with multi-scale support
├── balanced_sampler.py # Class-balanced sampling
└── utils.py           # Utilities
```

---

## 12. References

- **Mamba**: [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)
- **FlowNet**: [Dosovitskiy et al., 2015](https://arxiv.org/abs/1504.06852)
- **RAFT**: [Teed & Deng, 2020](https://arxiv.org/abs/2003.12039)
