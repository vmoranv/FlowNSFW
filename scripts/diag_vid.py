"""Diagnose why vid_688504385ab68 is missed at all resolutions."""
import sys, torch, json
from pathlib import Path
sys.path.insert(0, "src")

from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset

ck = torch.load("final.pt", map_location="cuda")
m = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
m.load_state_dict(ck["model"]); m.eval()

# This specific NSFW video
frames_dir = Path("/mnt/d/cumhub/flow-nsfw/test_fresh/NSFW/vid_688504385ab68")
frames = sorted([str(f) for f in frames_dir.glob("*.jpg")])[:30]

manifest = json.dumps([{
    "video_id": "test", "label": 1, "split": "test", "frames": frames,
}])
with open("/tmp/single_test.json", "w") as f:
    f.write(manifest)

for res in [128, 192, 256, 320, 384, 480, 640]:
    ds = VideoClipDataset("/tmp/single_test.json", clip_len=4, resolution=(res, res), split="test")
    if len(ds) == 0:
        print(f"{res}p: skipped")
        continue
    s = ds[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(s["frames"].unsqueeze(0).cuda())
    probs = torch.softmax(o["video_cls"], dim=-1)
    sfw, nsfw = probs[0, 0].item(), probs[0, 1].item()
    pred = "NSFW" if nsfw > 0.5 else "SFW"
    flow_val = o["flow_fwd"].abs().mean().item()
    print(f"{res}p: {pred}  SFW={sfw:.6f}  NSFW={nsfw:.6f}  flow={flow_val:.4f}  frames={s['frames'].shape}")

# Also test: different clip starting points
print("\n--- Testing different frame offsets ---")
for start in [0, 8, 15, 20]:
    sub_frames = frames[start:start+15]  # 15 consecutive frames
    if len(sub_frames) < 4:
        continue
    manifest2 = json.dumps([{
        "video_id": f"test_{start}", "label": 1, "split": "test", "frames": sub_frames,
    }])
    with open("/tmp/single_test2.json", "w") as f:
        f.write(manifest2)
    ds2 = VideoClipDataset("/tmp/single_test2.json", clip_len=4, resolution=(320, 320), split="test")
    if len(ds2) == 0:
        print(f"  offset={start}: skipped")
        continue
    s2 = ds2[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o2 = m(s2["frames"].unsqueeze(0).cuda())
    probs2 = torch.softmax(o2["video_cls"], dim=-1)
    print(f"  offset={start}: NSFW conf={probs2[0,1].item():.6f}")
