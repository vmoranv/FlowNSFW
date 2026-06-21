"""Memory layout optimization — channels_last for Tensor Core efficiency.

One-line change, 20-40% speedup on modern GPUs (Ampere/Ada).
"""

import torch
import torch.nn as nn


def optimize_memory_layout(model: nn.Module) -> nn.Module:
    """Convert model to channels_last memory format.

    Args:
        model: FlowNSFW or any CNN-based model.

    Returns:
        Model with optimized layout (in-place).

    Usage:
        model = FlowNSFW(...)
        model = optimize_memory_layout(model)
        # All Conv2d now use NHWC (channels_last) internally
    """
    model = model.to(memory_format=torch.channels_last)
    print("[memory] converted to channels_last (NHWC) for Tensor Core efficiency")
    return model


def benchmark_layout(model: nn.Module, input_shape: tuple, device="cuda", warmup=10, iters=50):
    """Compare NCHW vs NHWC speed."""
    import time

    model_nchw = model.to(device)
    model_nhwc = model.to(device, memory_format=torch.channels_last)

    x = torch.randn(*input_shape, device=device)
    x_nhwc = x.to(memory_format=torch.channels_last)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model_nchw(x)
            _ = model_nhwc(x_nhwc)

    # Benchmark NCHW
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model_nchw(x)
    torch.cuda.synchronize()
    time_nchw = (time.perf_counter() - t0) / iters * 1000

    # Benchmark NHWC
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model_nhwc(x_nhwc)
    torch.cuda.synchronize()
    time_nhwc = (time.perf_counter() - t0) / iters * 1000

    speedup = (time_nchw - time_nhwc) / time_nchw * 100

    print(f"\n{'='*60}")
    print("Memory Layout Benchmark")
    print(f"{'='*60}")
    print(f"Input shape: {input_shape}")
    print(f"NCHW (default):      {time_nchw:.2f} ms/iter")
    print(f"NHWC (channels_last): {time_nhwc:.2f} ms/iter")
    print(f"Speedup:              {speedup:+.1f}%")
    print(f"{'='*60}")

    return {"nchw": time_nchw, "nhwc": time_nhwc, "speedup_pct": speedup}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from flow_nsfw import FlowNSFW

    model = FlowNSFW(dim=128, temporal_backend="attention").eval()
    benchmark_layout(model, (1, 4, 3, 320, 320), device="cuda")
