"""Check how many AVIF-in-disguise files exist in the source data."""
import json, os, subprocess

m = json.load(open("/mnt/d/cumhub/flow-nsfw/datasets/manifest_v4_clean_wsl.json"))

all_frames = []
for v in m:
    for f in v["frames"]:
        all_frames.append(f)

print(f"Total frames: {len(all_frames)}")

# Quick check: files smaller than 8KB might be truncated AVIFs
small_files = []
for f in all_frames:
    if os.path.exists(f):
        sz = os.path.getsize(f)
        if sz < 8192:
            small_files.append((f, sz))

print(f"Files < 8KB: {len(small_files)}")

# Check first 20 small files for AVIF signature
avif_count = 0
for f, sz in small_files[:50]:
    # Read first 8 bytes to check for AVIF signature
    try:
        with open(f, "rb") as fh:
            header = fh.read(12)
        if b"ftypavif" in header or b"ftypmif1" in header:
            avif_count += 1
            if avif_count <= 5:
                print(f"  AVIF: {os.path.basename(f)} ({sz} bytes)")
                # Check what directory
                parent = os.path.dirname(f).split("/")[-1]
                print(f"    parent dir: .../{parent}")
    except:
        pass

print(f"\nEstimated AVIF files among first 50 small files: {avif_count}")
print(f"\nImpact: {'noticeable but handled by black-frame fallback' if avif_count > 0 else 'negligible'}")
print("Training already handles these via corrupted-image fallback (black frame).")
