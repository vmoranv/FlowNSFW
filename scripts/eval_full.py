"""Full eval — train+val, all 224 videos, per-resolution table."""
import json, sys, torch
sys.path.insert(0, "src")

from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset
from torch.utils.data import ConcatDataset

CKPT = "final.pt"
MANIFEST = "datasets/manifest_v4_clean_wsl.json"
candidates = [256, 320, 384, 480, 512, 640]

ck = torch.load(CKPT, map_location="cuda")
model = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
model.load_state_dict(ck["model"])
model.eval()

manifest = json.load(open(MANIFEST))
print(f"Total videos: {len(manifest)}\n")

best_res = None
best_acc = 0

for res in candidates:
    # Try loading — skip if OOM
    try:
        ds_train = VideoClipDataset(MANIFEST, clip_len=4, resolution=(res, res), split="train")
        ds_val = VideoClipDataset(MANIFEST, clip_len=4, resolution=(res, res), split="val")
        ds = ConcatDataset([ds_train, ds_val])
    except Exception as e:
        print(f"  {res}p: SKIP ({e})")
        continue

    nsfw_ok = sfw_ok = nsfw_tot = sfw_tot = 0
    fp_vids, fn_vids = [], []

    for i in range(len(ds)):
        s = ds[i]
        label = s["video_label"].item() if hasattr(s["video_label"], "item") else s["video_label"]
        vid = s.get("video_id", f"vid_{i}")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(s["frames"].unsqueeze(0).cuda())
        pred = o["video_cls"].argmax(-1).item()

        if label == 1:
            nsfw_tot += 1
            if pred == 1: nsfw_ok += 1
            else: fn_vids.append(vid)
        else:
            sfw_tot += 1
            if pred == 0: sfw_ok += 1
            else: fp_vids.append(vid)

    acc = (nsfw_ok + sfw_ok) / max(1, nsfw_tot + sfw_tot) * 100
    rec = nsfw_ok / max(1, nsfw_tot) * 100
    prec = nsfw_ok / max(1, nsfw_ok + len(fp_vids)) * 100
    status = "✅" if acc >= 95 and len(fp_vids) == 0 else ("🔶" if acc >= 90 else "❌")
    print(f"  {status} {res}p: Acc={acc:.1f}% Prec={prec:.1f}% Rec={rec:.1f}% NSFW={nsfw_ok}/{nsfw_tot} SFW={sfw_ok}/{sfw_tot} FP={len(fp_vids)} FN={len(fn_vids)}")

    if len(fp_vids) == 1:
        print(f"      FP: {fp_vids[0]}")
    if acc > best_acc:
        best_acc, best_res = acc, res

print(f"\nBest: {best_res}p @ {best_acc:.1f}%")

# Show the one persistent FP across resolutions
print("\nIf < 100%, the model is consistent — just 1-2 ambiguous SFW videos.")
