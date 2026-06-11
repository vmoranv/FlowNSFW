"""V10 FlowNSFW vs YOLOv11 v16_s — head-to-head NSFW recall comparison."""

import sys, time, json
from pathlib import Path

# === CONFIG ===
FLOW_CKPT = "/mnt/d/cumhub/flow-nsfw/final.pt"
YOLO_CKPT = "/mnt/d/cumhub/anti-nsfw-yolo/shitreport/01_PyTorch_Weights/v16_s_best.pt"
MANIFEST = "/mnt/d/cumhub/flow-nsfw/datasets/manifest_v4_clean_wsl.json"

import torch
sys.path.insert(0, "/mnt/d/cumhub/flow-nsfw/src")

# === Load FlowNSFW ===
from flow_nsfw import FlowNSFW
flow_ck = torch.load(FLOW_CKPT, map_location="cuda", weights_only=False)
flow = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
flow.load_state_dict(flow_ck["model"])
flow.eval()

# === Load YOLOv11 ===
from ultralytics import YOLO
yolo = YOLO(YOLO_CKPT)
print(f"FlowNSFW=7.85M | YOLO v16_s=~5M\n")

# === Get all videos ===
manifest = json.load(open(MANIFEST))
videos = []
for v in manifest:
    frames = [f for f in v.get("frames", [])]
    frames = [f for f in frames if Path(f).exists()]
    if len(frames) < 8: continue
    videos.append({
        "id": v.get("video_id", f"v_{len(videos)}"),
        "label": v.get("label"),
        "frames": frames,
    })

nsfw_videos = [v for v in videos if v["label"] == 1]
sfw_videos = [v for v in videos if v["label"] == 0]
print(f"NSFW: {len(nsfw_videos)}  SFW: {len(sfw_videos)}  Total: {len(videos)}\n")

# === Test: all NSFW videos ===
import cv2, numpy as np
from flow_nsfw.data import _read_img

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

results = {"flow_nsfw_ok": 0, "yolo_ok": 0, "flow_sfw_ok": 0, "yolo_sfw_ok": 0}

print("=" * 80)
print("  NSFW RECALL COMPARISON")
print("=" * 80)

for v in videos:
    label = v["label"]
    frames = v["frames"]
    vid = v["id"]
    label_str = "NSFW" if label == 1 else "SFW"

    # --- FlowNSFW: sliding window ---
    imgs = sorted([f for f in frames[:30] if Path(f).suffix.lower() in IMAGE_EXTS])
    if len(imgs) < 8:
        continue
    sample = _read_img(imgs[0])
    H, W = sample.shape[:2]
    max_dim = 384
    scale = max_dim / max(H, W) if max(H, W) > max_dim else 1.0
    H, W = int(H * scale), int(W * scale)
    H, W = (H + 7) // 8 * 8, (W + 7) // 8 * 8

    frame_data = np.zeros((len(imgs), H, W, 3), dtype=np.uint8)
    for i, p in enumerate(imgs):
        img = _read_img(p)
        if img is None: img = np.zeros((H, W, 3), dtype=np.uint8)
        if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H))
        frame_data[i] = img
    ft = torch.from_numpy(frame_data.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

    flow_nsfw = False
    for start in range(0, len(imgs) - 8 + 1, 4):
        clip = ft[start:start + 8].unsqueeze(0).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = flow(clip)
        if torch.softmax(o["video_cls"], dim=-1)[0, 1].item() > 0.5:
            flow_nsfw = True
            break

    flow_correct = (flow_nsfw == (label == 1))

    # --- YOLO: per-frame ---
    yolo_nsfw = False
    for p in imgs[:min(20, len(imgs))]:
        try:
            result = yolo(str(p), verbose=False)
            for r in result:
                if r.boxes is not None and len(r.boxes) > 0:
                    cls_ids = r.boxes.cls.cpu().numpy()
                    if (cls_ids > 0).any():  # class 0=SFW, 1-4=NSFW
                        yolo_nsfw = True
                        break
        except:
            pass
        if yolo_nsfw:
            break

    yolo_correct = (yolo_nsfw == (label == 1))

    flow_icon = "✅" if flow_correct else "❌"
    yolo_icon = "✅" if yolo_correct else "❌"
    flow_str = "NSFW" if flow_nsfw else "SFW"
    yolo_str = "NSFW" if yolo_nsfw else "SFW"

    print(f"  {vid:<25} GT:{label_str:>4}  Flow:{flow_icon} {flow_str:<4}  YOLO:{yolo_icon} {yolo_str:<4}")

    if label == 1:
        if flow_correct: results["flow_nsfw_ok"] += 1
        if yolo_correct: results["yolo_ok"] += 1
    else:
        if flow_correct: results["flow_sfw_ok"] += 1
        if yolo_correct: results["yolo_sfw_ok"] += 1

# Summary
n_total = len(nsfw_videos) + len(sfw_videos)
n_nsfw = len(nsfw_videos)
n_sfw = len(sfw_videos)
flow_acc = (results["flow_nsfw_ok"] + results["flow_sfw_ok"]) / n_total * 100
yolo_acc = (results["yolo_ok"] + results["yolo_sfw_ok"]) / n_total * 100
flow_nsfw_rec = results["flow_nsfw_ok"] / n_nsfw * 100
yolo_nsfw_rec = results["yolo_ok"] / n_nsfw * 100

print(f"\n{'='*60}")
print(f"{'':<20} {'FlowNSFW V10':>15} {'YOLOv11 v16_s':>15}")
print(f"{'Accuracy':<20} {flow_acc:>14.1f}% {yolo_acc:>14.1f}%")
print(f"{'NSFW Recall':<20} {flow_nsfw_rec:>14.1f}% {yolo_nsfw_rec:>14.1f}%")
print(f"{'NSFW Detected':<20} {results['flow_nsfw_ok']:>14}/{n_nsfw} {results['yolo_ok']:>14}/{n_nsfw}")
print(f"{'SFW Correct':<20} {results['flow_sfw_ok']:>14}/{n_sfw} {results['yolo_sfw_ok']:>14}/{n_sfw}")
