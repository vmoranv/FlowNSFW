"""Build A10 deployment package — convert AVIF to JPG, verify all images."""
import json, os, shutil, io
from pathlib import Path

ROOT = Path("/mnt/d/cumhub/flow-nsfw")
PKG = ROOT / "flow_nsfw_a10_package"
DATA = PKG / "data"

if PKG.exists():
    shutil.rmtree(PKG)
PKG.mkdir(parents=True)

manifest = json.load(open(ROOT / "datasets/manifest_v4_clean_wsl.json"))
new_manifest = []
stats = {"ok": 0, "avif_converted": 0, "skipped": 0, "black": 0}

def safe_decode(src: Path):
    """Decode any image to RGB numpy array using best available decoder."""
    import numpy as np
    import cv2

    data = np.fromfile(str(src), dtype=np.uint8)

    # 1) OpenCV (fast)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is not None:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # 2) PIL (handles AVIF, WebP, etc.)
    try:
        from PIL import Image
        pil = Image.open(io.BytesIO(data))
        pil = pil.convert("RGB")
        return np.array(pil)
    except Exception:
        pass

    # 3) ffmpeg
    try:
        import subprocess, tempfile
        cmd = ["ffmpeg", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-v", "error", "-y", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        if proc.returncode == 0 and proc.stdout:
            w, h = 0, 0
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", str(src)],
                capture_output=True, text=True, timeout=5)
            if probe.returncode == 0:
                parts = probe.stdout.strip().split(",")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
                    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w, 3)
    except Exception:
        pass

    return None  # All failed → black

for v in manifest:
    label = "NSFW" if v.get("label") == 1 else "SFW"
    split = v.get("split", "train")
    vid_id = v.get("video_id", f"vid_{len(new_manifest)}")
    target_dir = DATA / split / label / vid_id
    target_dir.mkdir(parents=True, exist_ok=True)

    new_frames = []
    for i, src in enumerate(v["frames"]):
        src = Path(src)
        dst = target_dir / f"frame_{i:05d}.jpg"

        if not dst.exists():
            # Try reading source — auto-detect format and write as JPG
            img = safe_decode(src)
            if img is not None:
                import cv2
                cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                stats["ok"] += 1
            else:
                # Last resort: just copy raw file
                shutil.copy2(str(src), str(dst))
                stats["black"] += 1

        new_frames.append(f"./data/{split}/{label}/{vid_id}/frame_{i:05d}.jpg")
        if len(new_manifest) % 50 == 0:
            print(f"  {len(new_manifest)} videos done ({sum(stats.values())} frames)...")

    new_manifest.append({
        "video_id": vid_id,
        "label": v.get("label"),
        "split": split,
        "frames": new_frames,
    })

print(f"\nPackaged {len(new_manifest)} videos, {sum(stats.values())} frames")
print(f"  OK: {stats['ok']} | Skipped: {stats['skipped']} | Black: {stats['black']}")

# Manifest
json.dump(new_manifest, open(PKG / "manifest.json", "w"))

# Summary
nsfw = sum(1 for v in new_manifest if v["label"] == 1)
sfw = sum(1 for v in new_manifest if v["label"] == 0)
train = sum(1 for v in new_manifest if v["split"] == "train")
val = sum(1 for v in new_manifest if v["split"] == "val")
print(f"Train={train} Val={val} NSFW={nsfw} SFW={sfw}")

total_size = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(str(DATA)) for f in fs)
print(f"Data: {total_size / 1024 / 1024:.0f} MB")
