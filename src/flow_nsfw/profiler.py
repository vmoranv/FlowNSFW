"""Hotspot tracing for FlowNSFW — profile FLOPs/latency per module.

Usage:
    from flow_nsfw.profiler import profile_model

    model = FlowNSFW(...)
    x = torch.randn(1, 4, 3, 320, 320).cuda()

    profile_model(model, x, trace_file="hotspot.json")
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class HotspotTracer:
    """Lightweight FLOPs + latency profiler per module."""

    def __init__(self):
        self.hooks = []
        self.stats = {}

    def register(self, model: nn.Module):
        """Register forward hooks on all modules."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # Estimate FLOPs (rough heuristic)
            flops = 0
            if isinstance(module, nn.Conv2d):
                in_c = module.in_channels
                out_c = module.out_channels
                k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                out_h = output.shape[2] if len(output.shape) == 4 else 1
                out_w = output.shape[3] if len(output.shape) == 4 else 1
                flops = in_c * out_c * k * k * out_h * out_w
            elif isinstance(module, nn.Linear):
                flops = module.in_features * module.out_features
            elif isinstance(module, nn.MultiheadAttention):
                # Q@K^T + softmax@V
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input
                B, L, D = inp.shape
                flops = 2 * L * L * D  # approximate

            # Latency measurement (sync GPU)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000

            # Accumulate
            if name not in self.stats:
                self.stats[name] = {"flops": 0, "latency_ms": 0.0, "count": 0}
            self.stats[name]["flops"] += flops
            self.stats[name]["latency_ms"] += latency_ms
            self.stats[name]["count"] += 1

        return hook

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def summary(self, top_k: int = 20) -> dict:
        """Return top-K hotspots by FLOPs."""
        items = sorted(self.stats.items(), key=lambda x: -x[1]["flops"])
        return dict(items[:top_k])


def profile_model(model: nn.Module, sample_input: Tensor, trace_file: str = None, top_k: int = 20):
    """Profile model and optionally save to JSON.

    Args:
        model: FlowNSFW instance.
        sample_input: (B, T, 3, H, W) tensor.
        trace_file: JSON output path.
        top_k: top K modules to report.

    Returns:
        dict: {module_name: {flops, latency_ms, count}}
    """
    tracer = HotspotTracer()
    tracer.register(model)

    model.eval()
    with torch.no_grad():
        _ = model(sample_input)

    tracer.cleanup()
    summary = tracer.summary(top_k=top_k)

    print(f"\n{'='*60}")
    print(f"Hotspot Trace (top {top_k})")
    print(f"{'='*60}")
    print(f"{'Module':<40} {'FLOPs (G)':<12} {'Latency (ms)':<12}")
    print(f"{'-'*60}")
    for name, stat in summary.items():
        print(f"{name:<40} {stat['flops']/1e9:<12.3f} {stat['latency_ms']:<12.3f}")

    if trace_file:
        Path(trace_file).write_text(json.dumps(summary, indent=2))
        print(f"\nTrace saved to {trace_file}")

    return summary


def compare_traces(trace1: dict, trace2: dict, name1="baseline", name2="optimized"):
    """Compare two traces and show improvements."""
    print(f"\n{'='*70}")
    print(f"Trace Comparison: {name1} vs {name2}")
    print(f"{'='*70}")
    print(f"{'Module':<40} {'Δ FLOPs (%)':<15} {'Δ Latency (%)':<15}")
    print(f"{'-'*70}")

    all_keys = set(trace1.keys()) | set(trace2.keys())
    for key in sorted(all_keys):
        s1 = trace1.get(key, {"flops": 0, "latency_ms": 0})
        s2 = trace2.get(key, {"flops": 0, "latency_ms": 0})

        flops_delta = ((s2["flops"] - s1["flops"]) / max(s1["flops"], 1)) * 100
        lat_delta = ((s2["latency_ms"] - s1["latency_ms"]) / max(s1["latency_ms"], 0.001)) * 100

        if abs(flops_delta) > 5 or abs(lat_delta) > 5:  # only show significant changes
            print(f"{key:<40} {flops_delta:<+15.1f} {lat_delta:<+15.1f}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, "src")
    from flow_nsfw import FlowNSFW

    model = FlowNSFW(dim=128, temporal_backend="attention").cuda().eval()
    x = torch.randn(1, 4, 3, 320, 320, device="cuda")

    trace = profile_model(model, x, trace_file="hotspot_baseline.json", top_k=15)
    print(f"\nTotal FLOPs: {sum(s['flops'] for s in trace.values())/1e9:.2f}G")
