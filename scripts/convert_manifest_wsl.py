"""Generate WSL-compatible manifest from Windows manifest."""
import json
import os
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "datasets/manifest_v3_with_real_sfw.json"
dst = src.replace(".json", "_wsl.json")

m = json.load(open(src))

for v in m:
    new_frames = []
    for f in v.get("frames", []):
        p = f.replace("D:\\", "/mnt/d/").replace("D:", "/mnt/d").replace("\\", "/")
        new_frames.append(p)
    v["frames"] = new_frames

json.dump(m, open(dst, "w"))
print(f"Converted {len(m)} entries -> {dst}")

# Check file existence
exist = sum(1 for v in m for f in v["frames"] if os.path.exists(f))
total = sum(len(v["frames"]) for v in m)
print(f"Files exist: {exist}/{total}")

if exist < total:
    missing = [f for v in m for f in v["frames"] if not os.path.exists(f)]
    print(f"Missing files ({len(missing)}):")
    for f in missing[:5]:
        print(f"  {f}")
