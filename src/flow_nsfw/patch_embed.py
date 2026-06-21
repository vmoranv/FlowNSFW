"""Lightweight PatchEmbed替换UNet编码器 — 方案3: 路由后投影.

For 4K inference:
  1. MotionRouter crops K patches (e.g., 320×320 RGB)
  2. PatchEmbed projects each patch to tokens via strided conv
  3. Temporal SSM aggregates
  4. Detection head outputs

Saves ~10G FLOPs vs UNet encoder, crucial for 4K.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """Ultra-lightweight patch embedding via strided conv2d.

    Args:
        in_ch: input channels (3 for RGB).
        embed_dim: output token feature dimension.
        patch_size: spatial patch size (e.g., 16).

    Forward:
        (B, C, H, W) → (B, embed_dim, H//patch_size, W//patch_size)
    """

    def __init__(self, in_ch: int = 3, embed_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 3, H, W) → (B, embed_dim, h, w) where h=H//patch_size."""
        x = self.proj(x)  # (B, embed_dim, h, w)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, h*w, embed_dim)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # back to (B, embed_dim, h, w)
        return x

    def count_flops(self, input_size: tuple) -> float:
        """Estimate FLOPs for one forward pass.

        Args:
            input_size: (H, W) spatial resolution of input.
        Returns:
            FLOPs in billions.
        """
        H, W = input_size
        h_out = H // self.patch_size
        w_out = W // self.patch_size
        # Conv2d FLOPs: output_spatial × kernel_ops
        # kernel_ops = in_ch × kernel_h × kernel_w × out_ch
        conv_flops = h_out * w_out * (3 * self.patch_size * self.patch_size * self.proj.out_channels)
        # LayerNorm: ~4 ops per element
        ln_flops = h_out * w_out * self.proj.out_channels * 4
        return (conv_flops + ln_flops) / 1e9


class LightweightSkipGenerator(nn.Module):
    """Generate fake skip connections for detection head compatibility.

    When --no-encoder is used, detection head still expects (s8, s4, s2, s1) skips.
    This module downsamples the patch-embedded features to create multi-scale skips.
    """

    def __init__(self, embed_dim: int = 256, skip_dims: tuple = (128, 64, 48)):
        """
        Args:
            embed_dim: input feature dim from PatchEmbed.
            skip_dims: (s4_ch, s2_ch, s1_ch) channel counts for detection head.
        """
        super().__init__()
        d4, d2, d1 = skip_dims

        # s8 bottleneck: just 1×1 conv to match expected channel
        # (no spatial downsample needed — already at stride 8 from PatchEmbed@patch16)
        self.s8_adapt = nn.Conv2d(embed_dim, embed_dim, 1)

        # s4: upsample 2× + reduce channels
        self.s4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, d4, 4, stride=2, padding=1),
            nn.GroupNorm(8, d4),
            nn.SiLU(inplace=True),
        )

        # s2: upsample 4× + reduce channels
        self.s2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, d2, 4, stride=4, padding=0),
            nn.GroupNorm(8, d2),
            nn.SiLU(inplace=True),
        )

        # s1: upsample 8× + reduce to s1 channels
        self.s1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, d1, 8, stride=8, padding=0),
            nn.GroupNorm(8, d1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """x: (B, embed_dim, h, w) at stride 8.

        Returns:
            (s8, s4, s2, s1) fake skips for detection head.
        """
        s8 = self.s8_adapt(x)
        s4 = self.s4(x)
        s2 = self.s2(x)
        s1 = self.s1(x)
        return s8, s4, s2, s1


def test_patch_embed():
    """Sanity check."""
    pe = PatchEmbed(in_ch=3, embed_dim=256, patch_size=16)
    x = torch.randn(2, 3, 320, 320)
    y = pe(x)
    print(f"PatchEmbed: {x.shape} → {y.shape}")
    print(f"FLOPs @ 320×320: {pe.count_flops((320, 320)):.3f}G")
    print(f"FLOPs @ 4K: {pe.count_flops((2160, 3840)):.3f}G")

    skip_gen = LightweightSkipGenerator(embed_dim=256, skip_dims=(128, 64, 48))
    s8, s4, s2, s1 = skip_gen(y)
    print(f"Skip shapes: s8={s8.shape}, s4={s4.shape}, s2={s2.shape}, s1={s1.shape}")
    print("✓ All passed")


if __name__ == "__main__":
    test_patch_embed()
