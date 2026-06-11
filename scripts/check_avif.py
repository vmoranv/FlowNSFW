import json, subprocess, os

m = json.load(open("/mnt/d/cumhub/flow-nsfw/datasets/manifest_v4_clean_wsl.json"))
for v in m:
    first_frame = v.get("frames", [""])[0]
    if "vid_99" in first_frame or "vid_53" in first_frame:
        print(f"video_id={v.get('video_id','?')}")
        print(f"  frames[0]={first_frame}")
        print(f"  total_frames={len(v['frames'])}")
        if os.path.exists(first_frame):
            result = subprocess.run(["file", first_frame], capture_output=True, text=True)
            print(f"  file type: {result.stdout.strip()}")
        print()
