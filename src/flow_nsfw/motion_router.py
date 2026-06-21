"""4K Motion Router — cheap frame-diff motion salience → RGB patch sampling.

Turns 4K inference from O(81×) infeasible to O(K patches) feasible.
All patches stay RGB (unified feature space); motion only routes WHERE to look.

Pipeline:
    4K frames (T,3,H,W)
      → motion_salience (frame-diff, on downsampled grid for 4K speed)
      → connectedComponents → top-K motion bboxes (normalized coords)
      → crop RGB patches at ORIGINAL resolution → resize to patch_size
      → global downsample fallback (catches static NSFW)
    Output: (K,T,3,ps,ps) RGB patches + (1,T,3,gr,gr) global + per-patch motion

The per-patch motion magnitude feeds the A4-硬切 (hard-switch) ablation; the
A4-软门 (soft-gate) ablation uses flow_fwd inside the model instead.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def motion_salience(frames: Tensor, blur: int = 5) -> Tensor:
    """Frame-difference motion salience.

    Args:
        frames: (T, 3, H, W) in [0,1].
        blur: avg-pool kernel for noise suppression (0 = off).
    Returns:
        salience: (T-1, H, W) in [0,1], per-frame max-normalized.
    """
    gray = frames.float().mean(dim=1)                  # (T, H, W)
    diff = (gray[1:] - gray[:-1]).abs()                # (T-1, H, W)
    if blur and blur > 1:
        diff = F.avg_pool2d(
            diff.unsqueeze(1), blur, stride=1, padding=blur // 2
        ).squeeze(1)
    mx = diff.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return diff / mx                                    # (T-1, H, W) in [0,1]


def extract_motion_bboxes(
    salience: Tensor,
    threshold: float = 0.15,
    min_area_frac: float = 0.002,
    max_patches: int = 8,
    dilate: int = 15,
) -> list[tuple[float, float, float, float]]:
    """Top-K motion bboxes from accumulated salience (normalized coords).

    Returns list of (x1, y1, x2, y2) in [0,1].
    """
    sal = salience.max(dim=0).values                    # (H, W) worst-frame motion
    mask = (sal > threshold).cpu().numpy().astype(np.uint8)
    H, W = mask.shape
    if mask.sum() == 0:
        return []
    if dilate > 0:
        k = max(3, dilate)
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cands = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area_frac * H * W:
            continue
        cands.append((x, y, x + w, y + h, area))
    cands.sort(key=lambda c: -c[4])
    cands = cands[:max_patches]

    min_side = max(W, H) // 6                           # enforce min patch size
    bboxes = []
    for x1, y1, x2, y2, _ in cands:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        if bw < min_side:
            x1 = max(0, int(cx - min_side / 2))
            x2 = min(W, int(cx + min_side / 2))
        if bh < min_side:
            y1 = max(0, int(cy - min_side / 2))
            y2 = min(H, int(cy + min_side / 2))
        bboxes.append((x1 / W, y1 / H, x2 / W, y2 / H))
    return bboxes


class MotionRouter:
    """4K motion router. All outputs are RGB — motion only selects WHERE.

    Args:
        patch_size: model input resolution patches are resized to.
        max_patches: cap on motion patches.
        threshold: salience binarization threshold.
        global_res: whole-frame downsample for static-content fallback.
        salience_res: salience computed on this grid (4K speedup); bbox
            coords are normalized so they map back to any source resolution.
        min_area_frac, dilate: connected-component filtering params.
    """

    def __init__(
        self,
        patch_size: int = 320,
        max_patches: int = 8,
        threshold: float = 0.15,
        global_res: int = 480,
        salience_res: int = 480,
        min_area_frac: float = 0.002,
        dilate: int = 15,
    ):
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.threshold = threshold
        self.global_res = global_res
        self.salience_res = salience_res
        self.min_area_frac = min_area_frac
        self.dilate = dilate

    @torch.no_grad()
    def __call__(self, frames: Tensor) -> dict:
        """Args:
            frames: (T, 3, H, W) in [0,1], any resolution incl. 4K.
        Returns:
            patches:       (K, T, 3, patch_size, patch_size) RGB
            patch_boxes:   list[(x1,y1,x2,y2)] normalized
            global_clip:   (1, T, 3, global_res, global_res) RGB fallback
            patch_motion:  (K,) per-patch motion magnitude
            global_motion: float, clip-level mean motion
            has_motion:    bool
        """
        T, _, H, W = frames.shape

        # --- salience on downsampled grid for 4K speed ---
        sal_frames = F.interpolate(
            frames, (self.salience_res, self.salience_res),
            mode="bilinear", align_corners=False,
        )                                              # (T,3,rs,rs)
        sal = motion_salience(sal_frames, blur=5)       # (T-1,rs,rs)
        global_motion = float(sal.mean().item())
        boxes = extract_motion_bboxes(
            sal, self.threshold, self.min_area_frac,
            self.max_patches, self.dilate,
        )                                               # normalized

        # --- crop RGB patches at ORIGINAL resolution ---
        patches, patch_motion = [], []
        for (nx1, ny1, nx2, ny2) in boxes:
            x1, y1 = int(round(nx1 * W)), int(round(ny1 * H))
            x2, y2 = int(round(nx2 * W)), int(round(ny2 * H))
            x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)
            patch = frames[:, :, y1:y2, x1:x2]          # (T,3,h,w)
            patch = F.interpolate(
                patch, (self.patch_size, self.patch_size),
                mode="bilinear", align_corners=False,
            )                                           # (T,3,ps,ps)
            pm = float(motion_salience(patch, blur=0).mean().item())
            patches.append(patch)
            patch_motion.append(pm)

        # --- global fallback (static content) ---
        g = F.interpolate(
            frames, (self.global_res, self.global_res),
            mode="bilinear", align_corners=False,
        )                                              # (T,3,gr,gr)

        if not patches:
            # no motion detected → carry global as the only patch
            patches = [F.interpolate(
                g, (self.patch_size, self.patch_size),
                mode="bilinear", align_corners=False,
            )]                                         # (T,3,ps,ps)
            patch_motion = [0.0]
            boxes = [(0.0, 0.0, 1.0, 1.0)]

        return {
            "patches": torch.stack(patches),            # (K,T,3,ps,ps)
            "patch_boxes": boxes,
            "global_clip": g.unsqueeze(0),              # (1,T,3,gr,gr)
            "patch_motion": torch.tensor(patch_motion),
            "global_motion": global_motion,
            "has_motion": len(boxes) > 0 and global_motion > 0.02,
            "n_patches": len(boxes),
        }


def frame_diff_input(frames: Tensor) -> Tensor:
    """Build a 3-channel motion-diff input for A4-硬切 (hard-switch) mode.

    (T,3,H,W) RGB → (T,3,H,W) where
        ch0 = |Δgray| forward,
        ch1 = |Δgray| backward,
        ch2 = gray (faint appearance hint so a 3-ch encoder still has signal).
    Pads to 3 channels so the shared RGB encoder accepts it without changing
    in_ch — this is the deliberate "feature-space mismatch" the hard-switch
    ablation is meant to expose (expect: static-content recall drops).
    """
    gray = frames.float().mean(dim=1, keepdim=True)     # (T,1,H,W)
    fwd = (gray[1:] - gray[:-1]).abs()
    bwd = (gray[:-1] - gray[1:]).abs()
    fwd = torch.cat([fwd, fwd[-1:]], dim=0)             # pad to length T
    bwd = torch.cat([bwd[:1], bwd], dim=0)
    return torch.cat([fwd, bwd, gray], dim=1)           # (T,3,H,W) in ~[0,1]
