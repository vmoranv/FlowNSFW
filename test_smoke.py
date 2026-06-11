import sys
sys.path.insert(0, "src")
import torch
from flow_nsfw.model import FlowNSFW

print("Building model...")
m = FlowNSFW(dim=96, flow_backend="scratch")
p = m.count_parameters()
print(f"Model OK: {p['total']/1e6:.2f}M params (encoder={p['encoder']/1e6:.2f}M flow={p['flow_net']/1e6:.2f}M temporal={p['temporal']/1e6:.2f}M)")

print("Forward pass...")
x = torch.randn(1, 4, 3, 320, 320)
o = m(x)
print(f"Forward OK")
print(f"  video_cls: {o['video_cls'].shape}")
print(f"  decoded scales: {len(o['decoded'])}")
print(f"  flow_fwd: {o['flow_fwd'].shape}")
for i, d in enumerate(o['decoded']):
    stride = d['stride']
    print(f"  scale s{stride}: cx={list(d['cx'].shape)} obj={list(d['obj'].shape)}")

print("\nAll good!")
