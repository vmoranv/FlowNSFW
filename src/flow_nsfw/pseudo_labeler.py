"""YOLO pseudo-labeler — convert video frames to FlowNSFW training manifest.

Handles AVIF frames by converting to numpy before YOLO inference.
Outputs JSON manifest consumable by VideoClipDataset.

Usage:
    python src/flow_nsfw/pseudo_labeler.py \
        --video-dirs D:/cumhub/anti-nsfw-yolo/data-collector/assets/pornhub \
        --yolo-model D:/cumhub/anti-nsfw-yolo/shitreport/01_PyTorch_Weights/v15_m_best.pt \
        --out datasets/manifest_nsfw.json --label 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Only needed for pseudo-labeling, not training
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def _decode_avif_to_numpy(path: Path) -> np.ndarray:
    """Convert AVIF to RGB numpy array via ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
        # Guess dimensions: AVIF stores width/height in container
        # Probe dimensions first
        probe = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=width,height",
                  "-of", "csv=p=0", str(path)]
        res = subprocess.run(probe, capture_output=True, timeout=10, text=True)
        parts = [x for x in res.stdout.strip().split(",") if x]
        w, h = int(parts[0]), int(parts[1])
        arr = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w, 3)
        return arr
    except Exception as e:
        raise RuntimeError(f"Cannot decode AVIF {path}: {e}")


def _decode_image(path: Path) -> np.ndarray:
    """Read any image as RGB numpy array.

    Decode chain: cv2 → PIL/pillow-avif → ffmpeg. Returns None only if all fail.
    """
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)

    # 1) Try OpenCV (fast path)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is not None:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # 2) Try PIL (handles AVIF if pillow-avif-plugin is installed)
    try:
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(data))
        pil_img = pil_img.convert("RGB")
        return np.array(pil_img)
    except Exception:
        pass

    # 3) Try ffmpeg for exotic / malformed formats
    try:
        return _decode_avif_to_numpy(path)
    except Exception:
        pass

    # All decoders failed — use black placeholder
    return None


def find_frame_dirs(root: Path, min_frames: int = 1) -> list[Path]:
    """Find leaf directories containing image sequences."""
    dirs = []
    for d in sorted(root.rglob("*")):
        if not d.is_dir():
            continue
        imgs = [f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if len(imgs) >= min_frames:
            dirs.append(d)
    return dirs


def label_video_frames(
    frame_dir: Path,
    yolo: YOLO,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cuda",
    frame_stride: int = 1,
    min_frames_for_clip: int = 2,
) -> dict:
    """Run YOLO on frames, decode AVIF→numpy as needed.

    For single-frame directories (e.g. VTS sprites), repeats the frame
    to create a clip-able entry.
    """
    imgs = sorted(
        [f for f in frame_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    )
    sampled = imgs[::frame_stride] if len(imgs) >= frame_stride else imgs
    frame_paths = [str(p.resolve()) for p in sampled]
    frame_labels = []
    detections = []

    for path in frame_paths:
        p = Path(path)
        try:
            if p.suffix.lower() == ".avif":
                arr = _decode_avif_to_numpy(p)
            else:
                arr = _decode_image(p)
            results = yolo(arr, device=device, imgsz=imgsz, conf=conf, verbose=False)
        except Exception as e:
            # Skip unreadable frames
            tqdm.write(f"  [skip] {p.name}: {e}")
            frame_labels.append(0)
            detections.append([])
            continue

        boxes = results[0].boxes
        has_nsfw = len(boxes) > 0
        frame_labels.append(int(has_nsfw))
        dets = []
        if has_nsfw:
            for box in boxes:
                dets.append({
                    "xyxy": box.xyxy[0].tolist(),
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                })
        detections.append(dets)

    # If single frame, duplicate to make a clip-able entry
    if len(sampled) < min_frames_for_clip:
        repeat = min_frames_for_clip // max(1, len(sampled))
        frame_paths = frame_paths * repeat
        frame_labels = frame_labels * repeat
        detections = detections * repeat

    return {
        "id": frame_dir.name,
        "frames": frame_paths,
        "frame_labels": frame_labels,
        "detections": detections,
        "n_nsfw_frames": sum(frame_labels),
        "n_frames": len(frame_paths),
    }


def build_manifest(
    video_roots: list[Path],
    yolo: YOLO,
    out_path: Path,
    label: int = 1,
    split: str = "train",
    val_ratio: float = 0.1,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cuda",
    frame_stride: int = 1,
    min_dir_frames: int = 1,
) -> None:
    """Build full manifest from video directories."""
    all_videos = []
    for root in video_roots:
        frame_dirs = find_frame_dirs(root, min_frames=min_dir_frames)
        print(f"[{root.name}] found {len(frame_dirs)} video directories")

        for fdir in tqdm(frame_dirs, desc=root.name):
            entry = label_video_frames(
                fdir, yolo, imgsz=imgsz, conf=conf,
                device=device, frame_stride=frame_stride,
            )
            entry["label"] = label
            all_videos.append(entry)

    if not all_videos:
        print("ERROR: no videos found!")
        return

    # Train/val split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_videos))
    n_val = max(1, int(len(all_videos) * val_ratio))
    val_idx = set(indices[:n_val].tolist())

    for i, v in enumerate(all_videos):
        v["split"] = "val" if i in val_idx else split

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_videos, f, indent=2, ensure_ascii=False)

    total_nsfw = sum(1 for v in all_videos if any(v["frame_labels"]))
    print(f"\nManifest saved: {out_path}")
    print(f"  Videos: {len(all_videos)} (train={len(all_videos)-n_val}, val={n_val})")
    print(f"  With NSFW frames: {total_nsfw}")
    print(f"  Total frames: {sum(v['n_frames'] for v in all_videos)}")


def main():
    ap = argparse.ArgumentParser(description="Build pseudo-labeled video manifest")
    ap.add_argument("--video-dirs", nargs="+", required=True,
                    help="Root directories containing video frame subdirs")
    ap.add_argument("--yolo-model", default="models/erax_nsfw_yolo11m.pt")
    ap.add_argument("--out", default="datasets/manifest.json")
    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--min-dir-frames", type=int, default=1,
                    help="Min image files per directory (1 for VTS sprites)")
    args = ap.parse_args()

    yolo = YOLO(args.yolo_model)
    build_manifest(
        video_roots=[Path(d) for d in args.video_dirs],
        yolo=yolo,
        out_path=Path(args.out),
        label=args.label,
        split=args.split,
        val_ratio=args.val_ratio,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        frame_stride=args.frame_stride,
        min_dir_frames=args.min_dir_frames,
    )


if __name__ == "__main__":
    main()
