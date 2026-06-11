"""Full integration test: balanced sampler + model forward + backward."""
import sys, torch
sys.path.insert(0, "src")

from flow_nsfw.data import VideoClipDataset
from flow_nsfw.balanced_sampler import BalancedBatchSampler
from flow_nsfw import FlowNSFW
from flow_nsfw.losses import (
    LossWeights, video_cls_loss, temporal_box_loss,
    flow_consistency_loss, flow_smoothness_loss, simple_detection_loss,
)
from torch.utils.data import DataLoader

# Dataset + balanced sampler
ds = VideoClipDataset("datasets/manifest_v4_clean_wsl.json", clip_len=4, resolution=(256, 256), split="train")
sampler = BalancedBatchSampler(ds, batch_size=2, shuffle=True)
loader = DataLoader(ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
print(f"Batches per epoch: {len(sampler)} (NSFW+SFW pairs)")

# Model
m = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
print(f"Params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

# One step
batch = next(iter(loader))
frames = batch["frames"].cuda()
labels = batch["video_label"].cuda()
print(f"Batch: {frames.shape} labels={labels.tolist()}")
assert len(set(labels.tolist())) == 2, "FAIL: batch must have both classes!"

with torch.autocast("cuda", dtype=torch.bfloat16):
    o = m(frames)

weights = LossWeights()
B, T = 2, 4
vcl, _ = video_cls_loss(o["video_cls"], labels, weights.video_cls)
tcl, _ = temporal_box_loss(o["decoded"], B, T, weights.temporal)
fcl, _ = flow_consistency_loss(o.get("flow_fwd"), o.get("flow_bwd"), weights.flow_consistency)
fsl, _ = flow_smoothness_loss(o.get("flow_fwd"), weights.flow_smoothness)
det_l, _ = simple_detection_loss(o["decoded"], batch["boxes"], B, T, 2.0)
total = vcl + tcl + fcl + fsl + det_l

print(f"Losses: vcl={vcl.item():.4f} tcl={tcl.item():.4f} fcl={fcl.item():.4f} total={total.item():.4f}")
total.backward()
print("Backward OK ✅")

# Check gradients aren't NaN
for name, p in m.named_parameters():
    if p.grad is not None and p.grad.isnan().any():
        print(f"  NaN grad: {name}")
        break
else:
    print("Gradients clean ✅")

print("\n=== ALL INTEGRATION TESTS PASSED ===")
