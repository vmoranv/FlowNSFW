"""FlowNSFW V10 vs YOLOv11 v16_s vs YOLOv11 auto_nsfw_v14 — 3-way benchmark."""
import sys, time, json
from pathlib import Path

import torch, cv2, numpy as np

# Models
FLOW_CKPT = "/mnt/d/cumhub/flow-nsfw/final.pt"
YOLO_S = "/mnt/d/cumhub/anti-nsfw-yolo/shitreport/01_PyTorch_Weights/v16_s_best.pt"
YOLO_AUTO = "/mnt/d/cumhub/anti-nsfw-yolo/models/auto_nsfw_v14.pt"
MANIFEST = "/mnt/d/cumhub/flow-nsfw/datasets/manifest_v4_clean_wsl.json"
OUT_MD = "/mnt/d/cumhub/flow-nsfw/BENCHMARK.md"

sys.path.insert(0, "/mnt/d/cumhub/flow-nsfw/src")

from flow_nsfw import FlowNSFW
from flow_nsfw.data import _read_img
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Load all 3 models
print("Loading models...")
flow = FlowNSFW(dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
                temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True).cuda()
flow.load_state_dict(torch.load(FLOW_CKPT, map_location="cuda", weights_only=False)["model"]); flow.eval()
yolo_s = YOLO(YOLO_S)
yolo_auto = YOLO(YOLO_AUTO)

manifest = json.load(open(MANIFEST))
videos = []
for v in manifest:
    frames = [f for f in v.get("frames", []) if Path(f).exists()]
    if len(frames) >= 8:
        videos.append({"id": v.get("video_id", f"v_{len(videos)}"), "label": v.get("label"), "frames": frames})

n_nsfw = sum(1 for v in videos if v["label"] == 1)
n_sfw = sum(1 for v in videos if v["label"] == 0)
print(f"NSFW={n_nsfw} SFW={n_sfw} Total={len(videos)}\n")

def infer_flow(frame_dir):
    imgs = sorted([f for f in frame_dir[:30] if Path(f).suffix.lower() in IMAGE_EXTS])
    if len(imgs) < 8: return None, 0, 0
    sample = _read_img(imgs[0])
    H, W = sample.shape[:2]
    scale = 384 / max(H, W) if max(H, W) > 384 else 1.0
    Hr, Wr = int(H*scale), int(W*scale)
    Hr, Wr = (Hr+7)//8*8, (Wr+7)//8*8
    frames = np.zeros((len(imgs), Hr, Wr, 3), dtype=np.uint8)
    for i, p in enumerate(imgs):
        img = _read_img(p)
        if img is None: img = np.zeros((Hr,Wr,3), dtype=np.uint8)
        if img.shape[:2] != (Hr,Wr): img = cv2.resize(img, (Wr,Hr))
        frames[i] = img
    ft = torch.from_numpy(frames.astype(np.float32)/255.0).permute(0,3,1,2)
    t0 = time.perf_counter()
    nsfw = False
    for start in range(0, len(imgs)-8+1, 4):
        clip = ft[start:start+8].unsqueeze(0).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = flow(clip)
        if torch.softmax(o["video_cls"], dim=-1)[0,1].item() > 0.5:
            nsfw = True; break
    return nsfw, time.perf_counter()-t0, 0

def infer_yolo(model, frame_dir):
    imgs = sorted([f for f in frame_dir[:30] if Path(f).suffix.lower() in IMAGE_EXTS])
    nsfw, elapsed = False, time.perf_counter()
    for p in imgs[:min(20, len(imgs))]:
        try:
            r = model(str(p), verbose=False)
            for det in r:
                if det.boxes is not None and len(det.boxes) > 0:
                    if (det.boxes.cls.cpu().numpy() > 0).any():
                        nsfw = True; break
        except: pass
        if nsfw: break
    return nsfw, time.perf_counter()-elapsed

# Run
results = {"flow": {"ok":0,"nsfw_ok":0,"sfw_ok":0,"time":0},
           "yolo_s": {"ok":0,"nsfw_ok":0,"sfw_ok":0,"time":0},
           "yolo_auto": {"ok":0,"nsfw_ok":0,"sfw_ok":0,"time":0}}

rows = []
for v in videos:
    label, vid, frames = v["label"], v["id"], v["frames"]
    gt = "NSFW" if label == 1 else "SFW"

    f_n, f_t, _ = infer_flow(frames)
    ys_n, ys_t = infer_yolo(yolo_s, frames)
    ya_n, ya_t = infer_yolo(yolo_auto, frames)

    for model_name, pred, elapsed in [("flow", f_n, f_t), ("yolo_s", ys_n, ys_t), ("yolo_auto", ya_n, ya_t)]:
        r = results[model_name]
        r["time"] += elapsed
        if (pred == (label == 1)): r["ok"] += 1
        if label == 1 and pred: r["nsfw_ok"] += 1
        if label == 0 and not pred: r["sfw_ok"] += 1

    fa = "✅" if f_n == (label==1) else "❌"
    ys = "✅" if ys_n == (label==1) else "❌"
    ya = "✅" if ya_n == (label==1) else "❌"
    rows.append((vid, gt, fa, "NSFW" if f_n else "SFW", f_t, ys, "NSFW" if ys_n else "SFW", ys_t, ya, "NSFW" if ya_n else "SFW", ya_t))

# Report
tot = n_nsfw + n_sfw
def rpt(name, r):
    acc = r["ok"]/tot*100
    rec = r["nsfw_ok"]/n_nsfw*100
    sfw = r["sfw_ok"]/n_sfw*100
    ms = r["time"]/tot*1000
    return f"| {name} | {acc:.1f}% | {rec:.1f}% ({r['nsfw_ok']}/{n_nsfw}) | {sfw:.1f}% ({r['sfw_ok']}/{n_sfw}) | {ms:.0f}ms |"

lines = []
L = lines.append
L("# FlowNSFW V10 vs YOLOv11 v16_s vs YOLOv11 auto_nsfw_v14\n")
L(f"**Date**: 2026-06-11 | **GPU**: RTX 5060 8GB | **Videos**: {tot} ({n_nsfw} NSFW + {n_sfw} SFW)\n")
L("## Summary\n")
L("| Model | Accuracy | NSFW Recall | SFW Accuracy | Avg Time |")
L("|-------|----------|-------------|--------------|----------|")
L(rpt("**FlowNSFW V10**", results["flow"]))
L(rpt("YOLOv11 v16_s", results["yolo_s"]))
L(rpt("YOLOv11 auto_nsfw_v14", results["yolo_auto"]))
L("")
L("## Per-Video\n")
L("| # | Video | GT | FlowNSFW | ms | YOLO S | ms | YOLO Auto | ms |")
L("|---|-------|----|----------|----|--------|----|-----------|----|")
for i,(vid,gt,fa,fs,ft,ysa,yss,yst,yaa,yas,yat) in enumerate(rows[:50]):
    L(f"| {i} | {vid[:15]} | {gt} | {fa} {fs} | {ft*1000:.0f} | {ysa} {yss} | {yst*1000:.0f} | {yaa} {yas} | {yat*1000:.0f} |")

with open(OUT_MD, "w") as f: f.write("\n".join(lines))

print(f"\n{'='*65}")
print(f"{'Model':<25} {'Acc':>7} {'NSFW Rec':>10} {'SFW Acc':>10} {'Avg ms':>8}")
print(f"{'FlowNSFW V10':<25} {results['flow']['ok']/tot*100:>6.1f}% {results['flow']['nsfw_ok']/n_nsfw*100:>9.1f}% {results['flow']['sfw_ok']/n_sfw*100:>9.1f}% {results['flow']['time']/tot*1000:>7.0f}")
print(f"{'YOLOv11 v16_s':<25} {results['yolo_s']['ok']/tot*100:>6.1f}% {results['yolo_s']['nsfw_ok']/n_nsfw*100:>9.1f}% {results['yolo_s']['sfw_ok']/n_sfw*100:>9.1f}% {results['yolo_s']['time']/tot*1000:>7.0f}")
print(f"{'YOLOv11 auto_nsfw_v14':<25} {results['yolo_auto']['ok']/tot*100:>6.1f}% {results['yolo_auto']['nsfw_ok']/n_nsfw*100:>9.1f}% {results['yolo_auto']['sfw_ok']/n_sfw*100:>9.1f}% {results['yolo_auto']['time']/tot*1000:>7.0f}")
print(f"\nReport: {OUT_MD}")
