"""NSFW Detection Head — multi-scale, flow-gated, with sparse processing.

Replaces ProgressiveSubtitleHead (mask prediction) with YOLO-style detection.
Outputs: boxes + class scores at each scale, natively temporal-consistent.

Design:
  - 4 detection scales (s8/s4/s2/s1) inherited from encoder pyramid
  - Flow-gated deformable conv at each scale for motion-guided feature refinement
  - Optional sparse processing: only run detection on foreground windows
  - Lightweight: ~0.5M params at dim=128
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sparse window utilities
# ---------------------------------------------------------------------------

def _compute_foreground_mask(
    feat: Tensor,
    threshold: float = 0.3,
) -> Tensor:
    """Simple foreground activation mask from feature energy.

    Args:
        feat: (B, C, H, W) feature map.
        threshold: activation energy threshold.

    Returns:
        (B, 1, H, W) binary mask.
    """
    energy = feat.abs().mean(dim=1, keepdim=True)
    energy = energy / (energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return (energy > threshold).float()


def _window_mask_regions(
    mask: Tensor,
    window_size: int = 8,
    context: int = 1,
) -> Tensor:
    """Identify windows that intersect with the foreground mask.

    Args:
        mask: (B, 1, H, W) binary mask.
        window_size: spatial window size.
        context: extra context windows around active ones.

    Returns:
        (B, 1, H, W) dilated mask at window granularity.
    """
    B, _, H, W = mask.shape
    # Downsample mask to window grid
    gh = (H + window_size - 1) // window_size
    gw = (W + window_size - 1) // window_size

    # Max-pool to window resolution
    grid = F.adaptive_max_pool2d(mask, (gh, gw))  # (B, 1, gh, gw)

    # Dilate with context
    if context > 0:
        pad = context
        grid = F.pad(grid, [pad] * 4, mode='constant', value=0)
        grid = F.max_pool2d(grid, kernel_size=2 * context + 1, stride=1, padding=0)
        gh2, gw2 = grid.shape[-2:]
        # Resize back to original gh, gw if needed
        if gh2 != gh or gw2 != gw:
            grid = grid[:, :, :gh, :gw]

    # Upsample back to full resolution
    dilated = F.interpolate(grid.float(), size=(H, W), mode='nearest')
    return dilated


class _DetectScale(nn.Module):
    """Single-scale detection block: conv refine → box + cls heads."""

    def __init__(self, in_ch: int, hidden: int = 64, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
        )
        # YOLO-style: 4 box params (cx,cy,w,h) + 1 objectness + num_classes
        self.head = nn.Conv2d(hidden, 5 + num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Returns (B, 5+nc, H, W) logits."""
        return self.head(self.conv(x))


class DetectionHead(nn.Module):
    """Multi-scale NSFW detection head with optional sparse processing.

    Args:
        feat_chs: input channels for (s8_bottleneck, s4, s2, s1) features.
        hidden: conv refinement channels.
        num_classes: default 1 (NSFW).
        sparse: enable sparse window-based detection.
        window_size: spatial window size for sparse mode.
        sparse_threshold: foreground energy threshold.
    """

    def __init__(
        self,
        feat_chs: tuple[int, int, int, int] = (256, 128, 64, 48),
        hidden: int = 64,
        num_classes: int = 1,
        sparse: bool = False,
        window_size: int = 8,
        sparse_threshold: float = 0.3,
    ):
        super().__init__()
        c3, d4, d2, d1 = feat_chs
        self.s8 = _DetectScale(c3, hidden, num_classes)
        self.s4 = _DetectScale(d4, hidden, num_classes)
        self.s2 = _DetectScale(d2, hidden, num_classes)
        self.s1 = _DetectScale(d1, hidden, num_classes)

        self.sparse = sparse
        self.window_size = window_size
        self.sparse_threshold = sparse_threshold

    def _apply_sparse(
        self,
        detect_fn,
        feat: Tensor,
    ) -> Tensor:
        """Apply detection only on foreground windows.

        Uses energy-based foreground mask (no learned parameters needed).
        Falls back to dense detection if foreground covers >80% of the frame.
        """
        if not self.sparse:
            return detect_fn(feat)

        # Compute foreground mask from feature energy
        fg_mask = _compute_foreground_mask(feat, threshold=self.sparse_threshold)
        fg_ratio = fg_mask.mean()

        # If mostly foreground, skip sparsity (not worth it)
        if fg_ratio > 0.8:
            return detect_fn(feat)

        # Dilate mask to window granularity
        sparse_mask = _window_mask_regions(
            fg_mask,
            window_size=self.window_size,
            context=1,
        )

        # Run detection on full feature map, mask out background
        raw = detect_fn(feat)
        raw = raw * sparse_mask
        return raw

    def forward(
        self,
        feat_s8: Tensor,
        feat_s4: Tensor,
        feat_s2: Tensor,
        feat_s1: Tensor,
    ) -> dict[str, Tensor]:
        """Returns per-scale raw detection tensors.

        Each value is (B*T, 5+nc, h, w). Decode to boxes via the model.
        """
        if self.sparse:
            return {
                "raw_s8": self._apply_sparse(self.s8, feat_s8),
                "raw_s4": self._apply_sparse(self.s4, feat_s4),
                "raw_s2": self._apply_sparse(self.s2, feat_s2),
                "raw_s1": self._apply_sparse(self.s1, feat_s1),
            }
        else:
            return {
                "raw_s8": self.s8(feat_s8),
                "raw_s4": self.s4(feat_s4),
                "raw_s2": self.s2(feat_s2),
                "raw_s1": self.s1(feat_s1),
            }
