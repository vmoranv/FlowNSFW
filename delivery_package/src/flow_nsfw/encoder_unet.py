"""4-stage U-Net encoder — adapted from FlowEraser."""

import torch.nn as nn
from torch import Tensor


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
        nn.SiLU(inplace=True),
    )


class UNetEncoder(nn.Module):
    """4-level conv encoder: stride 1→2→4→8, returns bottleneck + skips."""

    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 128,
        skip_ratios: tuple[float, float, float] = (0.25, 0.5, 1.0),
        bottleneck_ratio: float = 2.0,
    ):
        super().__init__()
        c0 = max(8, int(dim * skip_ratios[0]))
        c1 = max(8, int(dim * skip_ratios[1]))
        c2 = max(8, int(dim * skip_ratios[2]))
        c3 = max(8, int(dim * bottleneck_ratio))
        self.channels = (c0, c1, c2, c3)
        self.stem = _conv_block(in_ch, c0, stride=1)
        self.down1 = _conv_block(c0, c1, stride=2)
        self.down2 = _conv_block(c1, c2, stride=2)
        self.down3 = _conv_block(c2, c3, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        b = self.down3(s2)
        return b, [s2, s1, s0]
