"""Analyze dataset and write fixes."""
import json, sys

m = json.load(open("datasets/manifest_v3_with_real_sfw_wsl.json"))
train = [v for v in m if v.get("split") == "train"]
val = [v for v in m if v.get("split") == "val"]
nsfw_train = [v for v in train if v.get("label") == 1]
sfw_train = [v for v in train if v.get("label") == 0]
static_sfw = [v for v in sfw_train if "auto_v14" in str(v.get("frames", [""])[0])]
pexels_sfw = [v for v in sfw_train if "pexels" in str(v.get("frames", [""])[0]).lower()]

print(f"Train: {len(train)} ({len(nsfw_train)} NSFW + {len(sfw_train)} SFW)")
print(f"  NSFW: {len(nsfw_train)}")
print(f"  SFW auto_v14 (static frames): {len(static_sfw)}")
print(f"  SFW pexels (real video): {len(pexels_sfw)}")
print(f"Val: {len(val)} ({len([v for v in val if v.get('label')==1])} NSFW + {len([v for v in val if v.get('label')==0])} SFW)")

# Check static frame issue: auto_v14 SFW videos - how many unique frames?
for v in static_sfw[:3]:
    frames = v.get("frames", [])
    unique = len(set(frames))
    print(f"  Static SFW example: {v['video_id']} -> {len(frames)} frames, {unique} unique")

for v in pexels_sfw[:3]:
    frames = v.get("frames", [])
    unique = len(set(frames))
    print(f"  Pexels SFW example: {v['video_id']} -> {len(frames)} frames, {unique} unique")
