"""Build pseudo-labeled video manifest from existing frame directories.

Usage with anti-nsfw-yolo paths:
    python scripts/build_manifest.py \
        --nsfw-dirs data-collector/assets/pornhub \
                     data-collector/assets/pornhub_video \
                     data-collector/assets/hanime \
        --sfw-dirs  datasets/sfw_videos \
        --yolo-model models/erax_nsfw_yolo11m.pt \
        --out datasets/manifest.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flow_nsfw.pseudo_labeler import build_manifest
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Build flow-nsfw training manifest")
    ap.add_argument("--nsfw-dirs", nargs="*", default=[],
                    help="Directories with NSFW video frames")
    ap.add_argument("--sfw-dirs", nargs="*", default=[],
                    help="Directories with SFW video frames")
    ap.add_argument("--yolo-model", default="models/erax_nsfw_yolo11m.pt")
    ap.add_argument("--out", default="datasets/manifest.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame-stride", type=int, default=2,
                    help="Sample every Nth frame (2 = half framerate)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    yolo = YOLO(args.yolo_model)
    print(f"[manifest] YOLO model loaded: {args.yolo_model}")

    # Process NSFW sources (label=1)
    if args.nsfw_dirs:
        nsfw_roots = [Path(d) for d in args.nsfw_dirs if Path(d).exists()]
        if nsfw_roots:
            build_manifest(
                video_roots=nsfw_roots,
                yolo=yolo,
                out_path=Path(args.out),
                label=1,
                split="train",
                val_ratio=args.val_ratio,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                frame_stride=args.frame_stride,
            )

    # Process SFW sources (label=0)
    if args.sfw_dirs:
        sfw_roots = [Path(d) for d in args.sfw_dirs if Path(d).exists()]
        if sfw_roots:
            # Append SFW entries to same manifest
            print("[manifest] Adding SFW sources (label=0)")
            # For now, process separately and manual merge
            build_manifest(
                video_roots=sfw_roots,
                yolo=yolo,
                out_path=Path(str(args.out).replace(".json", "_sfw.json")),
                label=0,
                split="train",
                val_ratio=0.0,  # val split handled by nsfw side
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                frame_stride=args.frame_stride,
            )

    print(f"[manifest] Done. Output: {args.out}")


if __name__ == "__main__":
    main()
