"""Smoke test for FlowNSFW with Mamba SSM in WSL."""
import sys
sys.path.insert(0, "src")

import torch
from flow_nsfw.ssm_backend import SSM_BACKEND, HAS_MAMBA_SSM, create_ssm_layer

print(f"SSM backend: {SSM_BACKEND}, CUDA: {HAS_MAMBA_SSM}")

# Test SSM layer
ssm = create_ssm_layer(d_model=128, d_state=16).cuda()
y = ssm(torch.randn(1, 32, 128, device="cuda"))
print(f"SSM forward OK: {y.shape}")

# Full model
from flow_nsfw import FlowNSFW
m = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2,
    sparse_detect=True,
).cuda()
counts = m.count_parameters()
print(f"FlowNSFW(mamba+sparse): {counts['total']/1e6:.2f}M params")

# Forward pass
frames = torch.randn(1, 4, 3, 160, 160, device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(frames)
print(f"Forward OK: video_cls={out['video_cls'].shape}, flow={out['flow_fwd'].shape}")
print(f"Scales: {list(out['raw'].keys())}")
print("\n=== ALL WSL TESTS PASSED ===")
