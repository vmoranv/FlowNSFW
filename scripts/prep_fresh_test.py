"""Prepare fresh NSFW and SFW test videos for evaluation."""
import json, os, shutil, cv2, random
from pathlib import Path

ROOT = Path("D:/cumhub/flow-nsfw")
TEST_DIR = ROOT / "test_fresh"
MANIFEST_V4 = ROOT / "datasets" / "manifest_v4_clean_wsl.json"

# Load existing manifest to find videos we haven't used
existing = json.load(open(MANIFEST_V4))
existing_dirs = set()
for v in existing:
    for f in v["frames"]:
        existing_dirs.add(os.path.dirname(f))

print(f"Existing videos: {len(existing)}")
print(f"Unique directories: {len(existing_dirs)}")

# Search for fresh NSFW videos
NSFW_SOURCES = [
    Path("D:/cumhub/anti-nsfw-yolo/data-collector/assets"),
    Path("D:/cumhub/anti-nsfw-yolo/data-collector/extracted"),
]

fresh_nsfw = []
for src_root in NSFW_SOURCES:
    if not src_root.exists():
        continue
    for d in src_root.rglob("*"):
        if not d.is_dir():
            continue
        d_str = str(d).replace("\\", "/")
        # Check images
        imgs = sorted([f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
        if len(imgs) < 5:
            continue
        # Check not already in training data
        if d_str.replace("/mnt/d/", "D:/") in existing_dirs or str(d) in existing_dirs:
            continue
        fresh_nsfw.append({"path": d, "imgs": imgs[:30], "label": 1})

print(f"Fresh NSFW candidates: {len(fresh_nsfw)}")

# Search for fresh SFW videos from Pexels that were NOT included
SFW_DIR = ROOT / "datasets" / "sfw_videos" / "videos"
fresh_sfw = []
if SFW_DIR.exists():
    used_pexels = set()
    for v in existing:
        frames = v.get("frames", [])
        if frames and "pexels" in frames[0].lower():
            used_pexels.add(os.path.dirname(frames[0]).split("/")[-1])

    for d in sorted(SFW_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name in used_pexels:
            continue
        imgs = sorted([f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
        if len(imgs) >= 5:
            fresh_sfw.append({"path": d, "imgs": imgs[:20], "label": 0})

print(f"Fresh SFW candidates: {len(fresh_sfw)}")

# Select: take up to 10 NSFW + 10 SFW
random.seed(42)
n_nsfw = min(10, len(fresh_nsfw))
n_sfw = min(10, len(fresh_sfw))

selected_nsfw = fresh_nsfw[:n_nsfw]
selected_sfw = fresh_sfw[:n_sfw]

print(f"\nSelected: {len(selected_nsfw)} NSFW + {len(selected_sfw)} SFW")

# Build manifest
test_manifest = []
TEST_DIR.mkdir(exist_ok=True)

for item in selected_nsfw + selected_sfw:
    label = item["label"]
    label_str = "NSFW" if label else "SFW"
    folder_name = item["path"].name
    vid_id = f"fresh_{label_str}_{folder_name}"

    # Copy frames to test dir
    target = TEST_DIR / label_str / folder_name
    target.mkdir(parents=True, exist_ok=True)

    frames = []
    for img in item["imgs"][:30]:
        dst = target / img.name
        if not dst.exists():
            shutil.copy2(str(img), str(dst))
        frames.append(str(dst))

    test_manifest.append({
        "video_id": vid_id,
        "label": label,
        "split": "test",
        "frames": frames,
    })

json.dump(test_manifest, open(TEST_DIR / "fresh_manifest.json", "w"))
print(f"\nSaved manifest: {TEST_DIR / 'fresh_manifest.json'} ({len(test_manifest)} videos)")
print(f"Test data: {TEST_DIR}")

# Print paths for manual inspection
print("\n=== FRESH TEST VIDEOS ===")
for item in test_manifest:
    label = "NSFW" if item["label"] else "SFW"
    print(f"  [{label}] {item['video_id']}")
    print(f"        {item['frames'][0]}")
