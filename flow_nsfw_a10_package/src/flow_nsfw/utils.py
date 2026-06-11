"""Shared utilities — adapted from FlowEraser."""

import torch
import torch.nn.functional as F
from torch import Tensor


def warp(feat: Tensor, flow: Tensor) -> Tensor:
    """Bilinear grid_sample warp."""
    B, _, H, W = feat.shape
    device, dtype = feat.device, feat.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack((xs, ys), dim=0).unsqueeze(0)
    grid = base + flow
    grid_x = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)


def resize_flow_sequence(flow: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize pixel-unit flow to target (H, W), rescaling magnitudes."""
    B, Tm, _, H, W = flow.shape
    th, tw = size
    if (H, W) == (th, tw):
        return flow
    flat = F.interpolate(flow.flatten(0, 1), size=(th, tw), mode="bilinear", align_corners=False)
    flat = flat.clone()
    flat[:, 0] *= tw / max(W, 1)
    flat[:, 1] *= th / max(H, 1)
    return flat.unflatten(0, (B, Tm))
