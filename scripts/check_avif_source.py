"""Check AVIF source file integrity."""
import json, os, subprocess

m = json.load(open("/mnt/d/cumhub/flow-nsfw/datasets/manifest_v4_clean_wsl.json"))

# Find vid_99 and vid_53
for vid_name in ["vid_99", "vid_53"]:
    matches = [v for v in m if any(vid_name in f for f in v.get("frames",[]))]
    if not matches:
        continue

    v = matches[0]
    print(f"\n=== {vid_name}: {len(v['frames'])} frames ===")

    # Check problematic frames
    problem_patterns = ["0046_2", "0006", "0427"]
    problematic = [f for f in v["frames"] if any(p in f for p in problem_patterns)]

    for p in problematic[:5]:
        if not os.path.exists(p):
            print(f"  MISSING: {p}")
            continue

        size = os.path.getsize(p)
        result = subprocess.run(["file", p], capture_output=True, text=True)
        ftype = result.stdout.strip().split(":", 1)[1].strip()
        print(f"  {os.path.basename(p)}: {size} bytes, {ftype}")
