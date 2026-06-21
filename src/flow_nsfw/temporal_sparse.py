"""Sparse global temporal aggregator — with Mamba SSM backend.

Per-frame aggregation across: self + warped neighbours + top-K distant tokens.
Operates at stride-8 bottleneck for feasible token counts.

Supports 3 temporal backends:
  - "attention" (default): standard Transformer with F.scaled_dot_product_attention
  - "mamba": O(N) SSM via mamba-ssm CUDA kernels (3-tier fallback chain)
  - "hybrid": attention for local (self+warped) + Mamba for global (top-K)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import warp
from .ssm_backend import create_ssm_layer, SSM_BACKEND


def _topk_tokens(q: Tensor, kv: Tensor, k: int) -> Tensor:
    q_mean = q.mean(dim=1, keepdim=True)
    q_mean = F.normalize(q_mean, dim=-1)
    kv_n = F.normalize(kv, dim=-1)
    score = (q_mean @ kv_n.transpose(-1, -2)).squeeze(1)
    k = min(k, kv.shape[1])
    _, idx = score.topk(k, dim=-1)
    return kv.gather(1, idx.unsqueeze(-1).expand(-1, -1, kv.shape[-1]))


# ---------------------------------------------------------------------------
# Backend 1: Attention blocks (original)
# ---------------------------------------------------------------------------

class _AttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, q_feat: Tensor, kv_stack: Tensor) -> Tensor:
        B, N, C = q_feat.shape
        h = self.num_heads
        q = self.qkv(q_feat)[..., :C].reshape(B, N, h, C // h).transpose(1, 2)
        k = self.qkv(kv_stack)[..., C:2*C].reshape(B, -1, h, C // h).transpose(1, 2)
        v = self.qkv(kv_stack)[..., 2*C:].reshape(B, -1, h, C // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _AttnBlock(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        x = q + self.attn(self.norm1(q), self.norm1(kv))
        return x + self.mlp(self.norm2(x))


# ---------------------------------------------------------------------------
# Backend 2: Mamba SSM blocks
# ---------------------------------------------------------------------------

class _MambaBlock(nn.Module):
    """SSM-based temporal block: O(N) complexity for long sequences."""

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2, ssm_backend: str = "auto"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = create_ssm_layer(
            d_model=dim, d_state=d_state, d_conv=4, expand=expand, backend=ssm_backend,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, _kv_unused: Tensor = None) -> Tensor:
        """x: (B, N, C). _kv_unused kept for API compatibility."""
        h = self.norm(x)
        h = self.ssm(h)
        # Gated residual
        gate = torch.sigmoid(self.gate_proj(self.norm(x)))
        x = x + gate * h
        return x + self.mlp(self.norm2(x))


# ---------------------------------------------------------------------------
# Backend 3: Hybrid — attention for local, Mamba for global
# ---------------------------------------------------------------------------

class _HybridBlock(nn.Module):
    """Attention for local (self + warped neighbors), Mamba for global context."""

    def __init__(self, dim: int, num_heads: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        # Local attention path
        self.norm_local = nn.LayerNorm(dim)
        self.attn = _AttnBlock(dim, num_heads)

        # Global SSM path
        self.norm_global = nn.LayerNorm(dim)
        self.ssm = create_ssm_layer(
            d_model=dim, d_state=d_state, d_conv=4, expand=expand,
        )
        self.gate_proj = nn.Linear(dim, dim, bias=False)

        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # Step 1: Local attention (self + warped neighbors in kv)
        x = q + self.attn(self.norm_local(q), self.norm_local(kv))

        # Step 2: Global SSM pass over the result
        h = self.ssm(self.norm_global(x))
        gate = torch.sigmoid(self.gate_proj(self.norm_global(x)))
        x = x + gate * h

        # Step 3: MLP
        return x + self.mlp(self.norm_mlp(x))


# ---------------------------------------------------------------------------
# Top-level temporal aggregator
# ---------------------------------------------------------------------------

class SparseGlobalTemporal(nn.Module):
    """Temporal aggregator: window-local + sparse-global across clip frames.

    Args:
        dim: feature channels at stride-8.
        num_heads: attention heads (for attention/hybrid backends).
        num_layers: number of stacked blocks.
        topk: sparse-global token count per distant frame.
        temporal_backend: "attention" | "mamba" | "hybrid".
        d_state: SSM state size (for mamba/hybrid).
        ssm_expand: SSM expand factor (for mamba/hybrid).
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 6,
        num_layers: int = 3,
        topk: int = 64,
        temporal_backend: str = "attention",
        d_state: int = 16,
        ssm_expand: int = 2,
        motion_sparse_token: bool = False,
        sparse_topk: int = 200,
        ssm_backend: str = "auto",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.topk = topk
        self.backend = temporal_backend
        self.motion_sparse_token = motion_sparse_token
        self.sparse_topk = sparse_topk

        if temporal_backend == "mamba":
            self.blocks = nn.ModuleList([
                _MambaBlock(dim, d_state=d_state, expand=ssm_expand, ssm_backend=ssm_backend)
                for _ in range(num_layers)
            ])
            print(f"[temporal] backend=mamba (resolved: {SSM_BACKEND}, ssm_backend={ssm_backend}), "
                  f"d_state={d_state}, expand={ssm_expand}")
        elif temporal_backend == "hybrid":
            self.blocks = nn.ModuleList([
                _HybridBlock(dim, num_heads, d_state=d_state, expand=ssm_expand)
                for _ in range(num_layers)
            ])
            print(f"[temporal] backend=hybrid (attn+ssm via {SSM_BACKEND}), "
                  f"d_state={d_state}")
        else:
            self.blocks = nn.ModuleList([
                _TransformerBlock(dim, num_heads) for _ in range(num_layers)
            ])
            print(f"[temporal] backend=attention, heads={num_heads}")

    def _tokens(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2).contiguous(), (B, H, W)

    def _restore(self, x: Tensor, shape: tuple[int, int, int]) -> Tensor:
        B, H, W = shape
        return x.transpose(1, 2).reshape(B, -1, H, W)

    def _motion_topk_idx(self, anchor: Tensor, flow_fwd: Tensor,
                         t: int, T: int) -> tuple[Tensor, int]:
        """A3: pick top-K spatial positions by motion magnitude at frame t.

        anchor: (B,C,H,W). flow_fwd: (B,T-1,2,H,W) or None.
        Returns idx (B,K) into H*W flatten, and K.
        """
        B, C, H, W = anchor.shape
        if flow_fwd is not None and T > 1:
            fi = min(t, flow_fwd.shape[1] - 1)          # motion arriving at frame t
            mag = flow_fwd[:, fi].float().norm(dim=1)   # (B,H,W)
        else:
            mag = anchor.float().abs().mean(dim=1)      # fallback: feature energy
        mag = mag.view(B, H * W)
        K = min(self.sparse_topk, H * W)
        _, idx = mag.topk(K, dim=-1)                    # (B,K)
        return idx, K

    @staticmethod
    def _gather_tokens(x: Tensor, idx: Tensor) -> Tensor:
        """x: (B,C,H,W) → gather tokens at idx → (B,K,C)."""
        B, C, H, W = x.shape
        flat = x.flatten(2).transpose(1, 2)             # (B,HW,C)
        return flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))

    def _build_kv(
        self,
        feat: Tensor,
        flow_fwd: Tensor,
        t: int,
        T: int,
        q_tokens: Tensor,
    ) -> Tensor:
        """Build KV tokens: self + flow-warped neighbors + top-K global."""
        kv_parts: list[Tensor] = [q_tokens]
        if t > 0:
            kv_parts.append(self._tokens(warp(feat[:, t - 1], flow_fwd[:, t - 1]))[0])
        if t < T - 1:
            kv_parts.append(self._tokens(warp(feat[:, t + 1], -flow_fwd[:, t]))[0])
        if self.topk > 0:
            for s in range(T):
                if s in (t, t - 1, t + 1):
                    continue
                other, _ = self._tokens(feat[:, s])
                kv_parts.append(_topk_tokens(q_tokens, other, self.topk))
        return torch.cat(kv_parts, dim=1)

    def forward(self, feat: Tensor, flow_fwd: Tensor) -> Tensor:
        B, T, C, H, W = feat.shape
        out_frames: list[Tensor] = []

        for t in range(T):
            anchor = feat[:, t]

            if self.motion_sparse_token:
                # A3: only refine top-K motion positions; rest keep raw encoder feat.
                idx, K = self._motion_topk_idx(anchor, flow_fwd, t, T)
                q_tokens = self._gather_tokens(anchor, idx)        # (B,K,C)
                kv_all = self._build_kv(feat, flow_fwd, t, T, q_tokens)
                x = q_tokens
                for blk in self.blocks:
                    if isinstance(blk, _MambaBlock):
                        full_seq = torch.cat([kv_all, x], dim=1)
                        refined = blk(full_seq)
                        x = refined[:, kv_all.shape[1]:, :]
                    else:
                        x = blk(x, kv_all)
                # scatter refined tokens back to dense frame; keep raw feat elsewhere
                out_dense = anchor.clone()
                flat = out_dense.flatten(2)                        # (B,C,HW)
                flat.scatter_(2, idx.unsqueeze(1).expand(-1, C, -1),
                               x.transpose(1, 2))
                out_frames.append(flat.view(B, C, H, W))
                continue

            q_tokens, shape = self._tokens(anchor)
            kv_all = self._build_kv(feat, flow_fwd, t, T, q_tokens)

            x = q_tokens
            for blk in self.blocks:
                if isinstance(blk, _MambaBlock):
                    # Mamba is causal: put KV context BEFORE query so Q can see KV
                    # Sequence: [KV_self, KV_warped, KV_global | Q0...QN]
                    # The SSM scans left→right, so Q tokens see all KV tokens preceding them
                    full_seq = torch.cat([kv_all, x], dim=1)
                    refined = blk(full_seq)
                    x = refined[:, kv_all.shape[1]:, :]  # take back the query portion
                else:
                    # Attention or Hybrid: standard q + kv interface
                    x = blk(x, kv_all)

            out_frames.append(self._restore(x, shape))

        return torch.stack(out_frames, dim=1)
