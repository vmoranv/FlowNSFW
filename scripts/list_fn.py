"""List all FN (false negative) videos with their types and paths."""
import json, sys, torch
sys.path.insert(0, "src")

from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset

CKPT = "final.pt"
MANIFEST = "datasets/manifest_v4_clean_wsl.json"

ck = torch.load(CKPT, map_location="cuda")
model = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
model.load_state_dict(ck["model"])
model.eval()

# Use 256p which had best accuracy
ds = VideoClipDataset(MANIFEST, clip_len=4, resolution=(256, 256), split="train")

manifest = json.load(open(MANIFEST))
train_videos = [v for v in manifest if v.get("split") == "train"]

print("=== NSFW VIDEOS MISSED (False Negatives) ===\n")
fn_count = 0
for i in range(len(ds)):
    s = ds[i]
    label = s["video_label"].item() if hasattr(s["video_label"], "item") else s["video_label"]
    if label != 1:
        continue

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(s["frames"].unsqueeze(0).cuda())
    pred = o["video_cls"].argmax(-1).item()

    if pred == 0:  # Missed
        entry = train_videos[i] if i < len(train_videos) else {}
        vid = entry.get("video_id", "?")
        frames = entry.get("frames", [])
        first_frame = frames[0] if frames else "?"
        # Detect category from path
        category = "unknown"
        fp = first_frame.lower()
        if "anime2d" in fp: category = "anime2d"
        elif "iwara" in fp: category = "iwara"
        elif "render3d" in fp: category = "render3d"
        elif "semi2" in fp: category = "semi2d"
        elif "pornhub" in fp: category = "pornhub"
        elif "real" in fp: category = "real"

        probs = torch.softmax(o["video_cls"], dim=-1)
        conf_nsfw = probs[0, 1].item()
        conf_sfw = probs[0, 0].item()

        fn_count += 1
        # Windows path
        win_path = first_frame.replace("/mnt/d/", "D:/").replace("/", "\\")
        print(f"[{fn_count}] category={category}  vid={vid}")
        print(f"     NSFW conf={conf_nsfw:.4f}  SFW conf={conf_sfw:.4f}")
        print(f"     {win_path}")
        print()

print(f"Total missed: {fn_count}/124 NSFW videos")

# Also list SFW FPs
print("\n=== SFW VIDEOS WRONGLY FLAGGED (False Positives) ===\n")
fp_count = 0
for i in range(len(ds)):
    s = ds[i]
    label = s["video_label"].item() if hasattr(s["video_label"], "item") else s["video_label"]
    if label != 0:
        continue

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(s["frames"].unsqueeze(0).cuda())
    pred = o["video_cls"].argmax(-1).item()

    if pred == 1:  # False positive
        entry = train_videos[i] if i < len(train_videos) else {}
        vid = entry.get("video_id", "?")
        frames = entry.get("frames", [])
        first_frame = frames[0] if frames else "?"
        probs = torch.softmax(o["video_cls"], dim=-1)
        win_path = first_frame.replace("/mnt/d/", "D:/").replace("/", "\\")
        fp_count += 1
        print(f"[{fp_count}] vid={vid}")
        print(f"     NSFW conf={probs[0,1]:.4f}  SFW conf={probs[0,0]:.4f}")
        print(f"     {win_path}")
        print()

print(f"Total false positives: {fp_count}/100 SFW videos")
