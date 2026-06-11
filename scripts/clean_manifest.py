"""Generate clean manifest: drop static-frame SFW, keep only real-video SFW."""
import json

m = json.load(open("datasets/manifest_v3_with_real_sfw_wsl.json"))

# Split
train = [v for v in m if v.get("split") == "train"]
val = [v for v in m if v.get("split") == "val"]

# Filter: keep NSFW + pexels real-video SFW, drop auto_v14 static SFW
clean_train = []
dropped_static = 0
for v in train:
    frames = v.get("frames", [])
    # Check if all frames are same file (static repeat) or auto_v14
    is_static = len(set(frames)) == 1 or "auto_v14" in str(frames[0]) if frames else False
    if is_static and v.get("label") == 0:
        dropped_static += 1
        continue
    clean_train.append(v)

# Clean val too
clean_val = []
for v in val:
    frames = v.get("frames", [])
    is_static = len(set(frames)) == 1 or "auto_v14" in str(frames[0]) if frames else False
    if is_static and v.get("label") == 0:
        continue
    clean_val.append(v)

clean = clean_train + clean_val
print(f"Original: {len(train)} train + {len(val)} val = {len(m)} total")
print(f"Cleaned:  {len(clean_train)} train + {len(clean_val)} val = {len(clean)} total")
print(f"Dropped:  {dropped_static} static SFW from train, {len(val)-len(clean_val)} from val")

nsfw = sum(1 for v in clean if v.get("label") == 1)
sfw = sum(1 for v in clean if v.get("label") == 0)
print(f"NSFW: {nsfw}, SFW: {sfw} (ratio {nsfw}/{sfw}={nsfw/max(1,sfw):.1f}x)")

out = "datasets/manifest_v4_clean_wsl.json"
json.dump(clean, open(out, "w"))
print(f"\nSaved: {out}")
