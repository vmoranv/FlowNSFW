# FlowNSFW V8 — Mamba SSM NSFW Video Detection

## A10 Cloud Quick Start

```bash
cd delivery_package
bash train_v8_a10.sh
```

## Architecture

```
frames (B,T,3,H,W)
  → UNetEncoder → bottleneck + 3 skips
  → FlowNet (optimized correlation) → flow_fwd, flow_bwd
  → SparseGlobalTemporal (Mamba SSM, CUDA accelerated)
  → DetectionHead (4-scale, foreground-gated sparse)
  → VideoClassifier (Flow + RGB fusion)
```

## Key Features

- **Mamba SSM temporal aggregation** — O(N) vs O(N²) transformer
- **mamba_ssm CUDA kernels** — selective scan hardware acceleration
- **Balanced batch sampler** — 1 NSFW + 1 SFW per batch for contrastive learning
- **RGB + Flow fusion classifier** — sees both content AND motion
- **3-tier SSM fallback**: mamba_ssm → HF Mamba2 → cumprod fallback
- **7.85M params** — lightweight, fits on 8GB+ GPU

## Dataset

- 124 NSFW videos + 90 real SFW Pexels videos = 214 total
- Clean: zero static-frame samples (auto_v14 removed)
- Manifest: `datasets/manifest_v4_clean_wsl.json`

## Best Results (V7)

| Resolution | Accuracy | Recall | FP |
|-----------|----------|--------|-----|
| 160p | 87.5% | 100% | 2 |
| 480p | 68.8% | 100% | 5 |

V8 with clip_len=8 expected to improve further on A10.
