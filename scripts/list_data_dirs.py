"""List all data directories and compute total size for packaging."""
import json, os

m = json.load(open("datasets/manifest_v4_clean_wsl.json"))
all_files = []
for v in m:
    all_files.extend(v["frames"])

total_frames = len(all_files)
unique_dirs = sorted(set(os.path.dirname(f) for f in all_files))

print(f"Videos: {len(m)}")
print(f"Frames: {total_frames}")
print(f"Dirs:  {len(unique_dirs)}")
print()

total_size = 0
for d in unique_dirs:
    # Count files and size
    files_in_dir = [f for f in all_files if os.path.dirname(f) == d]
    size = 0
    for f in files_in_dir:
        if os.path.exists(f):
            size += os.path.getsize(f)
    total_size += size
    mb = size / 1024 / 1024
    print(f"  {d}  ({len(files_in_dir)} files, {mb:.1f} MB)")

print(f"\nTotal data size: {total_size / 1024 / 1024:.0f} MB")

# Check if files exist
missing = [f for f in all_files if not os.path.exists(f)]
if missing:
    print(f"\nWARNING: {len(missing)} missing files!")
    for f in missing[:5]:
        print(f"  MISSING: {f}")
else:
    print("\nAll files exist ✅")
