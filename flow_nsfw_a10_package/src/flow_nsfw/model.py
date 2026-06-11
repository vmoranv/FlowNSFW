"""FlowNSFW — Optical-flow-guided video NSFW detection with Mamba SSM support.

Pipeline:
    frames (B,T,3,H,W)
      → UNetEncoder(frames)       → bottleneck + 3 skips
      → FlowNet(bottleneck)       → flow_fwd, flow_bwd  (optimized correlation)
      → SparseGlobalTemporal      → feat_t (T,C,H/8,W/8)
          - backend="attention"  : standard Transformer (O(N²))
          - backend="mamba"      : SSM via mamba-ssm CUDA kernels (O(N))
          - backend="hybrid"     : attention(local) + SSM(global)
      → DetectionHead(feat_t, skips) → multi-scale raw detections
          - sparse=False : dense detection at all positions
          - sparse=True  : foreground-gated sparse window detection
      → decode_boxes()            → decoded [cx,cy,w,h, obj, cls] per scale

Training mode:
    - Detection loss (per-frame, per-scale) against pseudo-labeled GT
    - Video classification loss (optional)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder_unet import UNetEncoder
from .flow_net import FlowNet, RaftFlowNet
from .temporal_sparse import SparseGlobalTemporal
from .detection_head import DetectionHead
from .utils import resize_flow_sequence
from .ssm_backend import SSM_BACKEND


class FlowNSFW(nn.Module):
    """Video NSFW detection with optical flow and temporal modeling.

    Args:
        dim: encoder base channels
        num_heads: temporal attention heads
        num_temporal_layers: stacked temporal blocks
        topk_global: sparse-global token count per distant frame
        flow_backend: "scratch" or "raft"
        temporal_backend: "attention" | "mamba" | "hybrid"
        d_state: SSM state size (for mamba/hybrid backends)
        ssm_expand: SSM expand factor (for mamba/hybrid backends)
        sparse_detect: enable foreground-gated sparse detection
        num_classes: number of NSFW classes (default 1)
        detect_hidden: detection head conv channels
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_temporal_layers: int = 3,
        topk_global: int = 64,
        flow_backend: str = "scratch",
        temporal_backend: str = "attention",
        d_state: int = 16,
        ssm_expand: int = 2,
        sparse_detect: bool = False,
        num_classes: int = 1,
        detect_hidden: int = 64,
    ):
        super().__init__()
        self.flow_backend = flow_backend
        self.temporal_backend = temporal_backend

        # Encoder: bottleneck at stride 8
        self.encoder = UNetEncoder(
            in_ch=3, dim=dim,
            skip_ratios=(0.25, 0.5, 1.0),
            bottleneck_ratio=2.0,
        )
        c0, c1, c2, c3 = self.encoder.channels  # dim/4, dim/2, dim, dim*2

        # Flow estimator (optimized correlation via F.unfold + bmm)
        if flow_backend == "raft":
            self.flow_net = RaftFlowNet(feat_stride=8)
        else:
            self.flow_net = FlowNet(dim=c3)

        # Temporal aggregator at stride-8 bottleneck
        # Supports: "attention" | "mamba" | "hybrid"
        self.temporal = SparseGlobalTemporal(
            dim=c3, num_heads=num_heads,
            num_layers=num_temporal_layers, topk=topk_global,
            temporal_backend=temporal_backend,
            d_state=d_state, ssm_expand=ssm_expand,
        )

        # We use a simple conv decoder to upsample temporal features to all scales
        self.up_to_s4 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c2), nn.SiLU(inplace=True),
        )
        self.up_to_s2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),
            nn.GroupNorm(8, c1), nn.SiLU(inplace=True),
        )
        self.up_to_s1 = nn.Sequential(
            nn.ConvTranspose2d(c1, c0, 4, stride=2, padding=1),
            nn.GroupNorm(8, c0), nn.SiLU(inplace=True),
        )

        # Refine upsampled features with encoder skip connections
        self.fuse_s4 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, 3, padding=1),
            nn.GroupNorm(8, c2), nn.SiLU(inplace=True),
        )
        self.fuse_s2 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 3, padding=1),
            nn.GroupNorm(8, c1), nn.SiLU(inplace=True),
        )
        self.fuse_s1 = nn.Sequential(
            nn.Conv2d(c0 + c0, c0, 3, padding=1),
            nn.GroupNorm(8, c0), nn.SiLU(inplace=True),
        )

        # Detection head at 4 scales (optional sparse mode)
        self.detect = DetectionHead(
            feat_chs=(c3, c2, c1, c0),
            hidden=detect_hidden,
            num_classes=num_classes,
            sparse=sparse_detect,
        )

        # Temporal-to-appearance fusion for video classifier
        self.cls_fuse = nn.Sequential(
            nn.Conv2d(c3 + c0, c3, 3, padding=1),
            nn.GroupNorm(8, c3), nn.SiLU(inplace=True),
        )

        # Video-level classifier — temporal features + RGB appearance
        self.video_cls = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.GroupNorm(8, c3), nn.SiLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.GroupNorm(8, c3), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, num_classes + 2),
        )

    @torch.no_grad()
    def count_parameters(self) -> dict[str, int]:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        flow_train = sum(p.numel() for p in self.flow_net.parameters() if p.requires_grad)
        return {
            "encoder": n(self.encoder),
            "flow_net": n(self.flow_net),
            "flow_trainable": flow_train,
            "temporal": n(self.temporal),
            "decoder_upsample": n(self.up_to_s4) + n(self.up_to_s2) + n(self.up_to_s1),
            "decoder_fuse": n(self.fuse_s4) + n(self.fuse_s2) + n(self.fuse_s1),
            "detection_head": n(self.detect),
            "video_cls": n(self.video_cls),
            "total": n(self),
        }

    def _decode_predictions(
        self,
        raw: dict[str, Tensor],
        feat_hw: list[tuple[int, int]],
        imgsz: tuple[int, int],
    ) -> list[dict[str, Tensor]]:
        """Decode raw detection heads into boxes per scale."""
        scales = []
        strides = [8, 4, 2, 1]
        keys = ["raw_s8", "raw_s4", "raw_s2", "raw_s1"]
        H_img, W_img = imgsz

        for key, stride, (fh, fw) in zip(keys, strides, feat_hw):
            raw_tensor = raw[key]
            BxT, _, fh2, fw2 = raw_tensor.shape
            nc = raw_tensor.shape[1] - 5

            # Grid
            gy, gx = torch.meshgrid(
                torch.arange(fh2, device=raw_tensor.device, dtype=torch.float32),
                torch.arange(fw2, device=raw_tensor.device, dtype=torch.float32),
                indexing="ij",
            )
            grid_xy = torch.stack([gx, gy], dim=-1)

            # Decode
            box_raw = raw_tensor[:, :5]
            cls_raw = raw_tensor[:, 5:]

            cx = (torch.sigmoid(box_raw[:, 0]) + grid_xy[..., 0]) * stride / W_img
            cy = (torch.sigmoid(box_raw[:, 1]) + grid_xy[..., 1]) * stride / H_img
            w = torch.exp(box_raw[:, 2]) * stride / W_img
            h = torch.exp(box_raw[:, 3]) * stride / H_img
            obj = torch.sigmoid(box_raw[:, 4])
            cls_prob = torch.sigmoid(cls_raw)

            scales.append({
                "cx": cx, "cy": cy, "w": w, "h": h,
                "obj": obj, "cls": cls_prob,
                "stride": stride,
            })
        return scales

    def forward(
        self,
        frames: Tensor,
        cached_flow: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            frames: (B, T, 3, H, W) in [0,1].
            cached_flow: optional pre-computed (B, T-1, 2, H, W) flow.

        Returns:
            dict with:
              - raw: per-scale raw detection tensors
              - decoded: per-scale decoded boxes
              - video_cls: (B, nc+2) video-level logits
              - flow_fwd, flow_bwd
        """
        B, T, _, H, W = frames.shape

        # --- Encoder ---
        b_flat, skips_flat = self.encoder(frames.flatten(0, 1))
        c3 = b_flat.shape[1]
        b_seq = b_flat.unflatten(0, (B, T))  # (B,T,c3,H/8,W/8)

        # --- Flow (optimized correlation) ---
        if cached_flow is not None:
            flow_fwd = resize_flow_sequence(cached_flow, b_seq.shape[-2:])
            flow_bwd = -flow_fwd
        elif self.flow_backend == "raft":
            flow_fwd, flow_bwd = self.flow_net(b_seq, frames=frames)
        else:
            flow_fwd, flow_bwd = self.flow_net(b_seq)

        # --- Temporal aggregation (attention / mamba / hybrid) ---
        feat_t = self.temporal(b_seq, flow_fwd)  # (B,T,c3,H/8,W/8)
        feat_t_flat = feat_t.flatten(0, 1)       # (B*T,c3,H/8,W/8)

        # Split skips back to per-frame + flatten time
        skips_seq = [s.unflatten(0, (B, T)) for s in skips_flat]
        s2_seq, s1_seq, s0_seq = skips_seq
        s2_flat = s2_seq.flatten(0, 1)
        s1_flat = s1_seq.flatten(0, 1)
        s0_flat = s0_seq.flatten(0, 1)

        # --- Decode to multi-scale features ---
        f_s4 = self.fuse_s4(torch.cat([self.up_to_s4(feat_t_flat), s2_flat], dim=1))
        f_s2 = self.fuse_s2(torch.cat([self.up_to_s2(f_s4), s1_flat], dim=1))
        f_s1 = self.fuse_s1(torch.cat([self.up_to_s1(f_s2), s0_flat], dim=1))

        # --- Detection (dense or sparse) ---
        raw = self.detect(feat_t_flat, f_s4, f_s2, f_s1)

        feat_hw = [
            feat_t_flat.shape[-2:],
            f_s4.shape[-2:],
            f_s2.shape[-2:],
            f_s1.shape[-2:],
        ]
        decoded = self._decode_predictions(raw, feat_hw, (H, W))

        # --- Video classification (flow + RGB appearance) ---
        v_feat_flow = feat_t.mean(dim=1)  # (B,c3,H/8,W/8)
        # Fuse with RGB content from encoder skip (stride-1), downsample to match
        v_feat_rgb = s0_seq.mean(dim=1)   # (B,c0,H,W)
        v_feat_rgb_ds = F.adaptive_avg_pool2d(v_feat_rgb, v_feat_flow.shape[-2:])
        v_feat = self.cls_fuse(torch.cat([v_feat_flow, v_feat_rgb_ds], dim=1))
        v_cls = self.video_cls(v_feat)

        return {
            "raw": raw,
            "decoded": decoded,
            "video_cls": v_cls,
            "flow_fwd": flow_fwd,
            "flow_bwd": flow_bwd,
        }
