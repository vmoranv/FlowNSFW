"""Per-video evaluation with absolute paths — show every prediction."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset

# Config
CKPT = "runs/flow_nsfw_v4_mamba/final.pt"
MANIFEST = "datasets/manifest_v3_with_real_sfw_wsl.json"
RESOLUTIONS = [160, 320, 480, 640]

device = torch.device("cuda")
ck = torch.load(ROOT / CKPT, map_location=device)
model = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).to(device)
model.load_state_dict(ck["model"])
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, step={ck.get('step','?')}\n")

manifest = json.load(open(ROOT / MANIFEST))
val_entries = [v for v in manifest if v.get("split") == "val"]

for res in RESOLUTIONS:
    ds = VideoClipDataset(MANIFEST, clip_len=4, resolution=(res, res), split="val")
    print(f"\n{'='*80}")
    print(f"  RESOLUTION: {res}x{res}")
    print(f"{'='*80}")

    nsfw_correct, nsfw_total = 0, 0
    sfw_correct, sfw_total = 0, 0
    fp_list, fn_list = [], []

    for i in range(len(ds)):
        s = ds[i]
        label = s["video_label"]
        if hasattr(label, "item"):
            label = label.item()
        video_id = s.get("video_id", f"video_{i}")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(s["frames"].unsqueeze(0).to(device))

        pred = o["video_cls"].argmax(-1).item()
        probs = torch.softmax(o["video_cls"], dim=-1)
        confidence = probs[0, pred].item()

        # Find original path
        entry = val_entries[i] if i < len(val_entries) else {}
        frames = entry.get("frames", [])
        sample_path = frames[0] if frames else "unknown"
        # Convert back to Windows path for easy access
        win_path = sample_path.replace("/mnt/d/", "D:/").replace("/", "\\")

        label_str = "NSFW" if label == 1 else "SFW"
        pred_str = "NSFW" if pred == 1 else "SFW"
        match = "✅" if label == pred else "❌"

        print(f"  {match} [{label_str}→{pred_str}] conf={confidence:.3f} id={video_id}")
        print(f"      {win_path}")

        if label == 1:
            nsfw_total += 1
            if pred == 1:
                nsfw_correct += 1
            else:
                fn_list.append(video_id)
        else:
            sfw_total += 1
            if pred == 0:
                sfw_correct += 1
            else:
                fp_list.append(video_id)

    acc = (nsfw_correct + sfw_correct) / max(1, nsfw_total + sfw_total) * 100
    rec = nsfw_correct / max(1, nsfw_total) * 100
    prec = nsfw_correct / max(1, nsfw_correct + len(fp_list)) * 100
    print(f"\n  📊 Accuracy={acc:.1f}%  Precision={prec:.1f}%  Recall={rec:.1f}%")
    print(f"     NSFW: {nsfw_correct}/{nsfw_total}  SFW: {sfw_correct}/{sfw_total}  FP={len(fp_list)} FN={len(fn_list)}")
    if fp_list:
        print(f"     False Positives (SFW→NSFW): {fp_list}")
    if fn_list:
        print(f"     False Negatives (NSFW→SFW): {fn_list}")
