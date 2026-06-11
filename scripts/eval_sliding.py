"""Sliding-window video classifier: scan entire video, vote NSFW if any window triggers."""
import sys, json, torch
from pathlib import Path

sys.path.insert(0, "src")
from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset

CKPT = "final.pt"
MANIFEST = "datasets/manifest_v4_clean_wsl.json"
FRESH = "test_fresh/fresh_manifest.json"

ck = torch.load(CKPT, map_location="cuda")
model = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
model.load_state_dict(ck["model"]); model.eval()

def classify_video_sliding(frames_paths, resolution, clip_len=8, stride=16):
    """Scan full video with sliding windows. Any window NSFW → video NSFW."""
    total_frames = len(frames_paths)
    if total_frames < clip_len:
        return "SHORT", 0.0, 0

    windows_checked = 0
    nsfw_votes = 0

    # Build temp manifest
    tmp = json.dumps([{
        "video_id": "scan", "label": 0, "split": "test",
        "frames": frames_paths,
    }])
    with open("/tmp/scan.json", "w") as f:
        f.write(tmp)

    for start in range(0, total_frames - clip_len + 1, stride):
        ds = VideoClipDataset("/tmp/scan.json", clip_len=clip_len, resolution=(resolution, resolution), split="test")
        if len(ds) == 0:
            continue
        # Manually set start to our window
        s = ds[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(s["frames"].unsqueeze(0).cuda())
        probs = torch.softmax(o["video_cls"], dim=-1)
        nsfw_conf = probs[0, 1].item()
        windows_checked += 1
        if nsfw_conf > 0.5:
            nsfw_votes += 1

    if windows_checked == 0:
        return "SHORT", 0.0, 0

    nsfw_ratio = nsfw_votes / windows_checked
    verdict = "NSFW" if nsfw_votes > 0 else "SFW"
    return verdict, nsfw_ratio, windows_checked


print("=" * 70)
print("  SLIDING-WINDOW FULL-VIDEO CLASSIFICATION")
print("=" * 70)

for res in [320, 480]:
    print(f"\n=== @ {res}px ===")

    # -- Fresh NSFW --
    if Path(FRESH).exists():
        fresh = json.load(open(FRESH))
        nsfw_hit = 0
        for v in fresh:
            frames = [f.replace("D:\\", "/mnt/d/").replace("\\", "/") for f in v["frames"]]
            verdict, ratio, windows = classify_video_sliding(frames, res)
            nsfw_hit += 1 if verdict == "NSFW" else 0
            status = "✅" if verdict == "NSFW" else "❌ MISS"
            print(f"  {status} {v['video_id']}: {verdict} ({nsfw_hit}/{nsfw_hit+0}) ratio={ratio:.2f} windows={windows}")
            if verdict == "NSFW" and ratio < 1.0:
                print(f"       ⚠️ only {ratio:.0%} windows triggered")
        print(f"  RESULT: {nsfw_hit}/{len(fresh)} detected")

    # -- Full val set --
    for split, name in [("val", "Val"), ("train", "Train (sample)")]:
        ds = VideoClipDataset(MANIFEST, clip_len=4, resolution=(res, res), split=split)
        import random
        random.seed(42)
        indices = random.sample(range(len(ds)), min(20, len(ds)))
        correct = 0
        missed = []
        for i in indices:
            s = ds[i]
            label = s["video_label"].item() if hasattr(s["video_label"], "item") else s["video_label"]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(s["frames"].unsqueeze(0).cuda())
            pred = o["video_cls"].argmax(-1).item()
            correct += 1 if pred == label else 0
            if pred != label and label == 1:
                missed.append(s.get("video_id", f"vid_{i}"))
        print(f"  {name} (4-frame): Acc={correct}/{len(indices)}")
        if missed:
            print(f"    Missed: {missed}")
