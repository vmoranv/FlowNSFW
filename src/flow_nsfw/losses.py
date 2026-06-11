"""FlowNSFW loss functions.

Supports:
  - Detection loss (per-scale): box regression + objectness + classification
  - Video-level classification loss
  - Temporal consistency loss on boxes
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossWeights:
    box: float = 5.0        # CIoU box regression
    obj: float = 1.0        # Objectness BCE
    cls: float = 1.0        # Classification BCE
    video_cls: float = 5.0  # Video-level classification — must dominate
    temporal: float = 0.05   # Box temporal smoothness (reduced)
    flow_consistency: float = 0.1  # Forward-backward flow consistency (reduced)
    flow_smoothness: float = 0.02  # Spatial smoothness of flow (reduced)
    scale_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)


def _ciou_loss(pred_boxes: dict[str, Tensor],
               target_boxes: Tensor,
               target_obj: Tensor) -> tuple[Tensor, Tensor]:
    """CIoU box loss for positive samples.

    pred_boxes: dict with cx,cy,w,h each (B*T, fh, fw)
    target_boxes: (B*T, 5, max_objs)  [cx,cy,w,h,cls]  padded with -1
    target_obj: (B*T, fh, fw)  1 if grid cell has object, else 0
    """
    # Simple version: build target per grid cell from YOLO assignment
    # For now use a direct MSE on box params weighted by objectness
    # This is a simplified loss — full YOLO assignment can be added later
    return torch.tensor(0.0, device=pred_boxes["cx"].device), torch.tensor(0.0)


def detection_loss(
    decoded: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    weights: LossWeights,
) -> tuple[Tensor, dict[str, float]]:
    """Per-scale detection loss.

    Args:
        decoded: list of 4 dicts (s8,s4,s2,s1) with cx,cy,w,h,obj,cls.
        targets: list of 4 target dicts at corresponding scales.
        weights: loss weight config.

    Returns:
        total loss, per-term log dict.
    """
    device = decoded[0]["cx"].device
    total = torch.tensor(0.0, device=device)
    logs: dict[str, float] = {}

    for i, (pred, tgt, sw) in enumerate(zip(decoded, targets, weights.scale_weights)):
        prefix = f"s{8 // (2**i)}"
        obj_mask = tgt.get("obj", torch.zeros_like(pred["obj"]))

        # Objectness BCE
        L_obj = F.binary_cross_entropy(pred["obj"], obj_mask, reduction="mean")
        total = total + sw * weights.obj * L_obj
        logs[f"L_obj_{prefix}"] = float(L_obj.detach())

        # Classification BCE (on positive cells)
        if obj_mask.sum() > 0:
            cls_target = tgt.get("cls", torch.zeros_like(pred["cls"]))
            L_cls = F.binary_cross_entropy(
                pred["cls"] * obj_mask.unsqueeze(1),
                cls_target * obj_mask.unsqueeze(1),
                reduction="sum",
            ) / obj_mask.sum().clamp_min(1)
            total = total + sw * weights.cls * L_cls
            logs[f"L_cls_{prefix}"] = float(L_cls.detach())

        # Box regression (on positive cells)
        if obj_mask.sum() > 0:
            for k in ("cx", "cy", "w", "h"):
                if k in tgt:
                    diff = (pred[k] - tgt[k]) * obj_mask
                    L_box_k = (diff ** 2).sum() / obj_mask.sum().clamp_min(1)
                    total = total + sw * weights.box * L_box_k
            logs[f"L_box_{prefix}"] = float(
                sum(((pred[k] - tgt.get(k, pred[k])) * obj_mask) ** 2
                    for k in ("cx", "cy", "w", "h") if k in tgt
                ).sum() / obj_mask.sum().clamp_min(1)
            )

    logs["L_detection"] = float(total.detach())
    return total, logs


def video_cls_loss(
    video_logits: Tensor,
    video_labels: Tensor,
    weight: float,
) -> tuple[Tensor, float]:
    """Video-level cross-entropy.

    Args:
        video_logits: (B, nc+2) logits.
        video_labels: (B,) int labels (0=SFW, 1=NSFW).
        weight: scalar weight.
    """
    loss = F.cross_entropy(video_logits, video_labels)
    return weight * loss, float(loss.detach())


def temporal_box_loss(
    decoded: list[dict[str, Tensor]],
    B: int, T: int,
    weight: float,
) -> tuple[Tensor, float]:
    """Penalize abrupt box changes between adjacent frames.

    Args:
        decoded: list of 4 scale dicts, each key maps to (B*T, ...).
        B, T: batch and time dims.
        weight: scalar weight.
    """
    total = torch.tensor(0.0, device=decoded[0]["cx"].device)
    for scale_pred in decoded:
        for k in ("cx", "cy", "w", "h"):
            x = scale_pred[k].unflatten(0, (B, T))
            diff = (x[:, 1:] - x[:, :-1]) ** 2
            total = total + diff.mean()
    return weight * total, float(total.detach())


def simple_detection_loss(
    decoded: list[dict[str, Tensor]],
    gt_boxes: list[list[Tensor]],
    B: int, T: int,
    weight: float,
) -> tuple[Tensor, float]:
    """Simplified detection loss using GT boxes from YOLO pseudo-labels.

    Args:
        decoded: List of 4 scale dicts with cx,cy,w,h (B*T, fh, fw)
        gt_boxes: List of B lists, each containing T tensors of shape (n_boxes, 5) [cx,cy,w,h,cls]
        B, T: batch and time dims
        weight: scalar weight

    Returns:
        Weighted loss and raw loss value
    """
    if not gt_boxes or not any(any(len(b) > 0 for b in batch) for batch in gt_boxes):
        return torch.tensor(0.0), 0.0

    device = decoded[0]["cx"].device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    # Use the largest scale (s8) for simplicity
    scale = decoded[0]  # s8: lowest resolution, largest receptive field
    cx_pred = scale["cx"].unflatten(0, (B, T))  # (B, T, fh, fw)
    cy_pred = scale["cy"].unflatten(0, (B, T))
    w_pred = scale["w"].unflatten(0, (B, T))
    h_pred = scale["h"].unflatten(0, (B, T))

    _, _, fh, fw = cx_pred.shape

    for b in range(B):
        for t in range(T):
            if t >= len(gt_boxes[b]) or len(gt_boxes[b][t]) == 0:
                continue

            gt = gt_boxes[b][t].to(device)  # (n_boxes, 5)

            # Map GT boxes to grid cells
            for box in gt:
                cx_gt, cy_gt, w_gt, h_gt, _ = box
                # Find nearest grid cell
                gx = int(cx_gt * fw)
                gy = int(cy_gt * fh)
                gx = min(max(gx, 0), fw - 1)
                gy = min(max(gy, 0), fh - 1)

                # MSE loss on box params
                loss = (
                    (cx_pred[b, t, gy, gx] - cx_gt) ** 2 +
                    (cy_pred[b, t, gy, gx] - cy_gt) ** 2 +
                    (w_pred[b, t, gy, gx] - w_gt) ** 2 +
                    (h_pred[b, t, gy, gx] - h_gt) ** 2
                )
                total_loss = total_loss + loss
                count += 1

    if count > 0:
        total_loss = total_loss / count

    return weight * total_loss, float(total_loss.detach())


def flow_consistency_loss(
    flow_fwd: Tensor,
    flow_bwd: Tensor,
    weight: float,
) -> tuple[Tensor, float]:
    """Forward-backward flow consistency loss.

    Args:
        flow_fwd: (B, T-1, 2, H, W) forward flow
        flow_bwd: (B, T-1, 2, H, W) backward flow
        weight: scalar weight

    Returns:
        Weighted loss and raw loss value
    """
    if flow_fwd is None or flow_bwd is None:
        return torch.tensor(0.0), 0.0

    B, Tm1, _, H, W = flow_fwd.shape

    # Warp flow_bwd by flow_fwd to check if it matches -flow_fwd
    # grid: (B, Tm1, H, W, 2) normalized to [-1, 1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow_fwd.device, dtype=torch.float32),
        torch.linspace(-1, 1, W, device=flow_fwd.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([x, y], dim=-1).unsqueeze(0).unsqueeze(0).expand(B, Tm1, -1, -1, -1)

    # Normalize flow to grid space (always float32 for grid operations)
    factor = torch.tensor([W / 2, H / 2], device=flow_fwd.device, dtype=torch.float32)
    flow_fwd_norm = flow_fwd.float().permute(0, 1, 3, 4, 2) / factor
    warped_grid = grid + flow_fwd_norm

    # Warp flow_bwd
    warped_grid_flat = warped_grid.reshape(B * Tm1, H, W, 2)

    flow_bwd_warped = F.grid_sample(
        flow_bwd.float().reshape(B * Tm1, 2, H, W),
        warped_grid_flat,
        align_corners=False,
        padding_mode="border",
    ).reshape(B, Tm1, 2, H, W)

    # Consistency: flow_fwd + warped(flow_bwd) ≈ 0
    consistency_error = (flow_fwd.float() + flow_bwd_warped).abs().mean()

    return weight * consistency_error, float(consistency_error.detach())


def flow_smoothness_loss(
    flow: Tensor,
    weight: float,
) -> tuple[Tensor, float]:
    """Spatial smoothness of optical flow.

    Args:
        flow: (B, T-1, 2, H, W) optical flow
        weight: scalar weight

    Returns:
        Weighted loss and raw loss value
    """
    if flow is None:
        return torch.tensor(0.0), 0.0

    # Gradient in x and y directions
    grad_x = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).abs()
    grad_y = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).abs()

    smoothness = (grad_x.mean() + grad_y.mean()) / 2
    return weight * smoothness, float(smoothness.detach())

