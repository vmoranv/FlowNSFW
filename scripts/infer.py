"""FlowNSFW inference — sliding-window, boxes like YOLO, per-frame labels.

Usage:
    python infer.py --ckpt final.pt --source D:/frames/ --device cuda
    python infer.py --ckpt final.pt --source D:/frames/ --save-frames D:/out/
    python infer.py --ckpt final.pt --manifest datasets/manifest.json
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import cv2, numpy as np, torch
from flow_nsfw import FlowNSFW
from flow_nsfw.data import _read_img

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_videos(source: Path, min_frames: int = 8) -> list[Path]:
    imgs = [f for f in source.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    if len(imgs) >= min_frames:
        return [source]
    dirs = []
    for d in sorted(source.iterdir()):
        if d.is_dir() and len([f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS]) >= min_frames:
            dirs.append(d)
    return dirs


def draw_box(img, cx, cy, w, h, conf, cls_name="NSFW", color=(0, 0, 255)):
    """Draw YOLO-style bounding box on image."""
    ih, iw = img.shape[:2]
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw - 1, x2), min(ih - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{cls_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def infer_video(model, frame_dir, device, clip_len=8, stride=4, draw_boxes=False):
    """Full sliding-window video classification + detection boxes.

    Returns per-frame NSFW confidence + bounding boxes from detection head.
    """
    imgs = sorted(f for f in Path(frame_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS)
    n_frames = len(imgs)
    if n_frames < clip_len:
        return None

    # Read first frame for sizing
    sample = _read_img(imgs[0])
    orig_H, orig_W = sample.shape[:2]

    # Downsize for GPU
    vram = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
    max_dim = 640 if vram >= 20000 else 480 if vram >= 12000 else 384
    if max(orig_H, orig_W) > max_dim:
        scale = max_dim / max(orig_H, orig_W)
        H, W = int(orig_H * scale), int(orig_W * scale)
    else:
        H, W = orig_H, orig_W
    H, W = (H + 7) // 8 * 8, (W + 7) // 8 * 8
    effective_res = f"{W}x{H}"

    # Load all frames into memory
    all_frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    for i, p in enumerate(imgs):
        img = _read_img(p)
        if img is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        all_frames[i] = img

    frames_t = torch.from_numpy(all_frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

    per_frame_conf = [0.0] * n_frames
    window_results = []
    nsfw_windows = 0
    total_frames_nsfw = set()

    for start in range(0, n_frames - clip_len + 1, stride):
        clip = frames_t[start:start + clip_len].unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(clip)

        # --- Video classification ---
        probs = torch.softmax(out["video_cls"], dim=-1)
        nsfw_conf = probs[0, 1].item()

        if nsfw_conf > 0.5:
            nsfw_windows += 1
            for fi in range(start, start + clip_len):
                per_frame_conf[fi] = max(per_frame_conf[fi], nsfw_conf)
                total_frames_nsfw.add(fi)

    max_conf = max(per_frame_conf)
    verdict = "NSFW" if max_conf > 0.5 else "SFW"

    return {
        "verdict": verdict,
        "max_conf": max_conf,
        "nsfw_windows": nsfw_windows,
        "total_windows": len(window_results),
        "nsfw_frame_count": len(total_frames_nsfw),
        "infer_resolution": effective_res,
        "n_frames": n_frames,
        "per_frame_conf": per_frame_conf,
        "windows": window_results,
        "_imgs": imgs if draw_boxes else None,
        "_all_frames": all_frames if draw_boxes else None,
    }


def main():
    ap = argparse.ArgumentParser(description="FlowNSFW Video Classifier with Boxes")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source")
    ap.add_argument("--manifest")
    ap.add_argument("--output")
    ap.add_argument("--save-frames", help="Save annotated frames to dir")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--temporal-backend", default="mamba")
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--ssm-expand", type=int, default=2)
    ap.add_argument("--sparse-detect", action="store_true", default=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.source and not args.manifest:
        ap.error("Specify --source or --manifest")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    ck = torch.load(args.ckpt, map_location=device)
    model = FlowNSFW(
        dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
        temporal_backend=args.temporal_backend,
        d_state=args.d_state, ssm_expand=args.ssm_expand,
        sparse_detect=args.sparse_detect,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Model: step={ck.get('step','?')}, "
          f"{(sum(p.numel() for p in model.parameters())/1e6):.2f}M params\n")

    # Collect videos
    videos = []
    if args.source:
        for d in find_videos(Path(args.source)):
            videos.append({"id": d.name, "path": d, "label": None})
    if args.manifest:
        m = json.load(open(args.manifest))
        for v in m:
            frames = [f.replace("D:\\", "/mnt/d/").replace("\\", "/") for f in v.get("frames", [])]
            frames = [f for f in frames if Path(f).exists()]
            if len(frames) >= args.clip_len:
                videos.append({
                    "id": v.get("video_id", f"vid_{len(videos)}"),
                    "path": Path(frames[0]).parent,
                    "label": v.get("label"),
                })
    if not videos:
        print("No videos found!"); return

    all_results = []
    nsfw_ok = sfw_ok = nsfw_tot = sfw_tot = 0

    for vi, vid in enumerate(videos):
        t0 = time.time()
        result = infer_video(model, vid["path"], device,
                             clip_len=args.clip_len, stride=args.stride,
                             draw_boxes=bool(args.save_frames))

        if result is None:
            print(f"[{vi+1}/{len(videos)}] {vid['id']}: TOO_SHORT")
            continue

        result["video_id"] = vid["id"]
        el = time.time() - t0

        gts = ""
        if vid["label"] is not None:
            gt = "NSFW" if vid["label"] == 1 else "SFW"
            ok = "✅" if result["verdict"] == gt else "❌"
            gts = f" GT={gt} {ok}"
            if vid["label"] == 1:
                nsfw_tot += 1
                if result["verdict"] == "NSFW": nsfw_ok += 1
            else:
                sfw_tot += 1
                if result["verdict"] == "SFW": sfw_ok += 1

        ns = result["nsfw_windows"]
        tw = result["total_windows"]
        nfc = result["nsfw_frame_count"]
        print(f"[{vi+1}/{len(videos)}] {vid['id']}: {result['verdict']:5s} "
              f"max_conf={result['max_conf']:.3f}  "
              f"nsfw_frames={nfc}/{result['n_frames']}  "
              f"nsfw_windows={ns}/{tw}  "
              f"{result['infer_resolution']}  {el:.1f}s{gts}")

        # Print per-window detail if mixed
        if 0 < ns < tw:
            nsfw_ranges = []
            for w in result["windows"]:
                if w["verdict"] == "NSFW":
                    nsfw_ranges.append(f"#{w['start']}-{w['end']}")
            if nsfw_ranges:
                ids = ", ".join(nsfw_ranges[:5])
                if len(nsfw_ranges) > 5:
                    ids += f" (+{len(nsfw_ranges)-5} more)"
                print(f"       NSFW ranges: {ids}")

        # Save annotated frames with boxes
        if args.save_frames:
            out_dir = Path(args.save_frames) / vid["id"]
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save keyframes: all NSFW frames + every Nth frame as thumbnail
            nsfw_frame_set = {i for i, c in enumerate(result["per_frame_conf"]) if c > 0.5}
            samples = set(nsfw_frame_set)
            for i in range(0, result["n_frames"], max(1, result["n_frames"] // 20)):
                samples.add(i)

            for fi in sorted(samples | nsfw_frame_set):
                if fi >= result["n_frames"]:
                    continue

                img = result["_all_frames"][fi].copy()
                conf = result["per_frame_conf"][fi]

                # Draw conf bar at bottom
                h, w = img.shape[:2]
                bar_h = 6
                bar_w = int(conf * w)
                bar_color = (0, 0, 255) if conf > 0.5 else (0, 200, 0)
                cv2.rectangle(img, (0, h - bar_h), (w, h), (200, 200, 200), -1)
                cv2.rectangle(img, (0, h - bar_h), (bar_w, h), bar_color, -1)

                # Label: "NSFW 0.95 | #20-28 #24-32"
                if conf > 0.5:
                    # Find windows covering this frame
                    win_ranges = []
                    for w in result["windows"]:
                        if w["verdict"] == "NSFW" and w["start"] <= fi < w["end"]:
                            win_ranges.append(f"#{w['start']}-{w['end']}")
                    win_str = " ".join(win_ranges[:3]) if win_ranges else ""
                    label = f"NSFW {conf:.2f} | {win_str}"
                    text_color = (255, 0, 0)
                else:
                    label = f"SAFE {1-conf:.2f}"
                    text_color = (0, 180, 0)
                cv2.putText(img, label, (4, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                out_path = out_dir / f"f{fi:05d}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        del result["_imgs"], result["_all_frames"]
        all_results.append(result)

    # Summary
    print(f"\n{'='*50}")
    nsfw_p = sum(1 for r in all_results if r["verdict"] == "NSFW")
    print(f"Total: {len(all_results)}  NSFW: {nsfw_p}")
    if nsfw_tot + sfw_tot > 0:
        acc = (nsfw_ok + sfw_ok) / (nsfw_tot + sfw_tot) * 100
        rec = nsfw_ok / max(1, nsfw_tot) * 100
        prec = nsfw_ok / max(1, nsfw_ok + (sfw_tot - sfw_ok)) * 100
        print(f"Acc={acc:.1f}%  Prec={prec:.1f}%  Rec={rec:.1f}%")
        print(f"NSFW: {nsfw_ok}/{nsfw_tot}  SFW: {sfw_ok}/{sfw_tot}")

    if args.output:
        for r in all_results:
            r.pop("windows", None)
            r.pop("per_frame_conf", None)
            r.pop("per_frame_boxes", None)
        json.dump(all_results, open(args.output, "w"), indent=2)
        print(f"Results → {args.output}")


if __name__ == "__main__":
    main()
