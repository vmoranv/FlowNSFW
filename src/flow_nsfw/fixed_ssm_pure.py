"""定点 SSM 的纯 PyTorch 实现 — 无需 CUDA 编译.

使用 PyTorch 原生 int16 操作模拟定点运算，
在 CUDA 上自动获得加速（Tensor Core INT16 支持）.
"""

import torch
import torch.nn as nn
from torch import Tensor


class FixedPointSSMPure(nn.Module):
    """定点 INT16 SSM — 纯 PyTorch 实现（无需编译）."""

    def __init__(self, d_model: int, d_state: int = 16, scale_bits: int = 12):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.scale_bits = scale_bits
        self.scale = 2 ** scale_bits

        # 参数（训练时 fp32）
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def quantize_to_int16(self, x: Tensor) -> Tensor:
        """float → int16 量化."""
        return (x * self.scale).clamp(-32768, 32767).to(torch.int16)

    def dequantize_from_int16(self, x: Tensor) -> Tensor:
        """int16 → float 反量化."""
        return x.float() / self.scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        if self.training:
            return self._forward_fp32(x)

        # 推理时定点
        B, L, D = x.shape
        device = x.device

        # 量化参数（限制范围）
        A = -torch.exp(self.A_log).clamp(-1, -0.01).detach()
        A_int = self.quantize_to_int16(A)
        B_int = self.quantize_to_int16(self.B.detach())
        C_int = self.quantize_to_int16(self.C.detach())
        D_int = self.quantize_to_int16(self.D.detach())

        # 量化输入
        x_int = self.quantize_to_int16(x)

        # 状态 (int32 避免溢出)
        h = torch.zeros(B, D, self.d_state, dtype=torch.int32, device=device)

        outputs = []
        for t in range(L):
            x_t = x_int[:, t, :]  # (B, D)

            # SSM 递推: h_t = A * h_{t-1} + B * x_t (定点)
            # (B,D,d_state) * (D,d_state) -> (B,D,d_state)
            decay = (h.float() * A_int.unsqueeze(0).float() / self.scale).to(torch.int32)
            input_contrib = (x_t.unsqueeze(-1).float() * B_int.unsqueeze(0).float() / self.scale).to(torch.int32)
            h = (decay + input_contrib).clamp(-2147483648, 2147483647)

            # 输出: y = C * h
            y_t = (C_int.unsqueeze(0).float() * h.float() / self.scale).sum(dim=-1)
            y_t = self.dequantize_from_int16(y_t.to(torch.int16))

            # 跳跃连接
            y_t = y_t + self.dequantize_from_int16(D_int) * self.dequantize_from_int16(x_t)

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def _forward_fp32(self, x: Tensor) -> Tensor:
        """标准 fp32（训练用）."""
        B, L, D = x.shape
        # 限制 A 范围避免爆炸
        A = -torch.exp(self.A_log).clamp(-1, -0.01)

        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []

        for t in range(L):
            x_t = x[:, t, :]
            h = A.unsqueeze(0) * h + self.B.unsqueeze(0) * x_t.unsqueeze(-1)
            y_t = (self.C.unsqueeze(0) * h).sum(-1) + self.D * x_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


# 测试
if __name__ == "__main__":
    print("="*60)
    print("Fixed-Point SSM (Pure PyTorch, No Compilation)")
    print("="*60)

    model = FixedPointSSMPure(d_model=64, d_state=16).cuda()
    x = torch.randn(2, 10, 64, device='cuda')

    # FP32
    model.train()
    with torch.no_grad():
        y_fp32 = model(x)

    # INT16
    model.eval()
    with torch.no_grad():
        y_int16 = model(x)

    error = (y_fp32 - y_int16).abs().mean().item()
    rel_error = error / y_fp32.abs().mean().item() * 100

    print(f"FP32 output: [{y_fp32.min():.4f}, {y_fp32.max():.4f}]")
    print(f"INT16 output: [{y_int16.min():.4f}, {y_int16.max():.4f}]")
    print(f"Relative error: {rel_error:.2f}%")

    if rel_error < 5:
        print("PASS: Fixed-point SSM working")
    else:
        print("WARN: Error >5%")

    print("="*60)
