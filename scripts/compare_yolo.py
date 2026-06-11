"""Head-to-head comparison: FlowNSFW V10 vs YOLO v16_s on the same videos."""
import json, sys, torch, cv2, numpy as np, time
from pathlib import Path

# Paths
FLOW_CKPT = "/mnt/d/cumhub/flow-nsfw/final.pt"
YOLO_CKPT = "/mnt/d/cumhub/anti-nsfw-yolo/shitreport/01_PyTorch_Weights/v16_s_best.pt"
MANIFEST = "/mnt/d/cumhub/flow-nsfw/datasets/manifest_v4_clean_wsl.json"
FRESH_MANIFEST = "/mnt/d/cumhub/flow-nsfw/test_fresh/fresh_manifest.json"

sys.path.insert(0, "/mnt/d/cumhub/flow-nsfw/src")

# ====== Load FlowNSFW ======
print("Loading FlowNSFW V10...")
from flow_nsfw import FlowNSFW
from flow_nsfw.data import _read_img

flow_ck = torch.load(FLOW_CKPT, map_location="cuda")
flow_model = FlowNSFW(
    dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
    temporal_backend="mamba", d_state=16, ssm_expand=2, sparse_detect=True,
).cuda()
flow_model.load_state_dict(flow_ck["model"])
flow_model.eval()
print(f"  FlowNSFW: step={flow_ck.get('step','?')}, 7.85M params")

# ====== Load YOLO ======
print("Loading YOLOv11 v16_s...")
yolo_ck = torch.load(YOLO_CKPT, map_location="cuda", weights_only=False)
yolo_model = yolo_ck["model"].float().eval()
print(f"  YOLO: epoch={yolo_ck.get('epoch','?')}, nc=5")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ====== Inference functions ======
def flow_infer_video(frame_dir, clip_len=8, stride=4):
    """Sliding-window FlowNSFW inference. Any window NSFW → video NSFW."""
    imgs = sorted([f for f in Path(frame_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS])
    if len(imgs) < clip_len:
        return {"verdict": "TOO_SHORT", "nsfw_frames": 0, "total_frames": len(imgs)}

    # Load + resize
    sample = _read_img(imgs[0])
    H, W = sample.shape[:2]
    max_dim = 384
    scale = max_dim / max(H, W) if max(H, W) > max_dim else 1.0
    H, W = int(H * scale), int(W * scale)
    H, W = (H + 7) // 8 * 8, (W + 7) // 8 * 8

    frames = np.zeros((len(imgs), H, W, 3), dtype=np.uint8)
    for i, p in enumerate(imgs):
        img = _read_img(p)
        if img is None: img = np.zeros((H, W, 3), dtype=np.uint8)
        if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H))
        frames[i] = img
    frames_t = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

    nsfw_frames = set()
    nsfw_windows = 0
    total_windows = 0

    for start in range(0, len(imgs) - clip_len + 1, stride):
        clip = frames_t[start:start + clip_len].unsqueeze(0).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = flow_model(clip)
        probs = torch.softmax(o["video_cls"], dim=-1)
        conf = probs[0, 1].item()
        total_windows += 1
        if conf > 0.5:
            nsfw_windows += 1
            for fi in range(start, start + clip_len):
                nsfw_frames.add(fi)

    verdict = "NSFW" if nsfw_windows > 0 else "SFW"
    return {
        "verdict": verdict, "nsfw_frames": len(nsfw_frames),
        "total_frames": len(imgs),
        "nsfw_windows": nsfw_windows, "total_windows": total_windows,
    }

def yolo_infer_video(frame_dir):
    """YOLO per-frame inference. Any frame with NSFW det → video NSFW."""
    imgs = sorted([f for f in Path(frame_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        return {"verdict": "TOO_SHORT", "nsfw_frames": 0, "total_frames": 0}

    nsfw_count = 0
    # NSFW classes in v16_s: class 0=SFW, class 1+ = NSFW variants
    # Default YOLO logit: argmax over classes → if class > 0, it's NSFW
    for p in imgs[:min(50, len(imgs))]:  # sample up to 50 frames for speed
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(640 / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.cuda()
        with torch.no_grad():
            preds = yolo_model(x)
        # preds is a list of DetectionModel outputs
        # For NSFW detection, we check if any detections have class > 0
        if isinstance(preds, (list, tuple)):
            for det in preds:
                if det is not None and hasattr(det, 'shape') and det.shape[-1] >= 6:
                    classes = det[:, -1].long()
                    if (classes > 0).any():
                        nsfw_count += 1
                        break

    verdict = "NSFW" if nsfw_count > 0 else "SFW"
    return {
        "verdict": verdict, "nsfw_frames": nsfw_count,
        "total_frames": min(50, len(imgs)),
    }

# ====== Run comparison ======
def load_videos(manifest_path):
    m = json.load(open(manifest_path))
    videos = []
    for v in m:
        frames = [f.replace("D:\\", "/mnt/d/").replace("\\", "/") for f in v.get("frames", [])]
        frames = [f for f in frames if Path(f).exists()]
        if len(frames) >= 8:
            videos.append({
                "id": v.get("video_id", f"v_{len(videos)}"),
                "label": v.get("label"),
                "path": Path(frames[0]).parent,
                "n_frames": len(frames),
            })
    return videos

all_videos = []
# Training+val set
all_videos.extend(load_videos(MANIFEST))
# Fresh videos
if Path(FRESH_MANIFEST).exists():
    all_videos.extend(load_videos(FRESH_MANIFEST))

print(f"\nComparing on {len(all_videos)} videos...\n")
print(f"{'Video':<30} {'GT':>4} {'FlowNSFW':>10} {'YOLO':>8} | {'Flow Detail':>20} | {'YOLO Detail':>20}")
print("-" * 120)

flow_correct = 0
yolo_correct = 0
flow_nsfw_correct = 0
yolo_nsfw_correct = 0
flow_sfw_correct = 0
yolo_sfw_correct = 0
nsfw_total = 0
sfw_total = 0
total = 0

for vi, v in enumerate(all_videos):
    label = v["label"]
    if label is None:
        continue
    total += 1
    if label == 1: nsfw_total += 1
    else: sfw_total += 1

    # FlowNSFW inference
    t0 = time.time()
    flow_result = flow_infer_video(v["path"])
    flow_time = time.time() - t0

    # YOLO inference
    t0 = time.time()
    yolo_result = yolo_infer_video(v["path"])
    yolo_time = time.time() - t0

    gt_str = "NSFW" if label == 1 else "SFW"

    # Status
    flow_ok = "✅" if flow_result["verdict"] == gt_str else "❌"
    yolo_ok = "✅" if yolo_result["verdict"] == gt_str else "❌"

    flow_detail = f"NSFW:{flow_result['nsfw_frames']}/{flow_result['total_frames']} win:{flow_result.get('nsfw_windows','?')}/{flow_result.get('total_windows','?')}"
    yolo_detail = f"NSFW:{yolo_result['nsfw_frames']}/{yolo_result['total_frames']}"

    print(f"{v['id'][:28]:<30} {gt_str:>4} {flow_result['verdict'] + flow_ok:>10} "
          f"{yolo_result['verdict'] + yolo_ok:>8} | {flow_detail:>20} | {yolo_detail:>20}")

    if flow_result["verdict"] == gt_str:
        flow_correct += 1
        if label == 1: flow_nsfw_correct += 1
        else: flow_sfw_correct += 1
    if yolo_result["verdict"] == gt_str:
        yolo_correct += 1
        if label == 1: yolo_nsfw_correct += 1
        else: yolo_sfw_correct += 1

# Summary
print(f"\n{'='*60}")
print(f"{' ':<30} {'FlowNSFW':>15} {'YOLO v16_s':>15}")
print(f"{'Accuracy':<30} {flow_correct/total*100:>14.1f}% {yolo_correct/total*100:>14.1f}%")
print(f"{'NSFW Recall':<30} {flow_nsfw_correct/nsfw_total*100:>14.1f}% {yolo_nsfw_correct/nsfw_total*100:>14.1f}%")
print(f"{'SFW Accuracy':<30} {flow_sfw_correct/sfw_total*100:>14.1f}% {yolo_sfw_correct/sfw_total*100:>14.1f}%")
print(f"{'NSFW Detected':<30} {flow_nsfw_correct:>14}/{nsfw_total} {yolo_nsfw_correct:>14}/{nsfw_total}")
print(f"{'SFW Correct':<30} {flow_sfw_correct:>14}/{sfw_total} {yolo_sfw_correct:>14}/{sfw_total}")
