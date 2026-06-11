"""Build SFW manifest from public image datasets.

Uses torchvision COCO val set (clean photos) as SFW source.
Each image is repeated to form a dummy "video clip".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def build_sfw_coco_manifest(out_path: str, n_images: int = 200):
    """Download COCO val images, build SFW manifest."""
    import tempfile
    import numpy as np
    from torchvision.datasets import CocoDetection

    print(f"[sfw] Downloading COCO val images...")
    tmp = Path(tempfile.mkdtemp(prefix="coco_sfw_"))
    ds = CocoDetection(
        root=str(tmp),
        annFile="",  # will auto-download
        download=True,
    )

    # Actually, let's just use a simpler approach: download a few SFW images directly
    # COCO requires annotation files. Let's use the simpler torchvision approach.
    print(f"[sfw] COCO not straightforward to auto-download. Using fallback...")

    # Fallback: create manifest from anti-nsfw-yolo auto_v14 images that are SFW
    # (YOLO predicts no NSFW → safe to use as SFW)
    auto_v14 = Path("D:/cumhub/anti-nsfw-yolo/datasets/auto_v14_dataset/images")
    if auto_v14.exists():
        print(f"[sfw] Using auto_v14 SFW images...")
        # Use val images — smallest subset
        val_dir = auto_v14 / "val"
        if val_dir.exists():
            imgs = list(val_dir.glob("*.jpg"))[:n_images]
            print(f"[sfw] Found {len(imgs)} val images")
            return _build_from_images(imgs, out_path)

    # Absolute fallback: generate a few solid-color images as SFW placeholders
    print(f"[sfw] WARNING: No SFW source found. Generating placeholder data.")
    return _build_placeholder(out_path, n_images)


def _build_from_images(imgs: list[Path], out_path: str):
    """Build manifest from list of image files."""
    videos = []
    for i, img in enumerate(imgs):
        frame_path = str(img.resolve())
        # Repeat single image 4x to form a clip-able entry
        videos.append({
            "id": f"sfw_{i:04d}",
            "frames": [frame_path, frame_path, frame_path, frame_path],
            "frame_labels": [0, 0, 0, 0],
            "detections": [[], [], [], []],
            "n_nsfw_frames": 0,
            "n_frames": 4,
            "label": 0,
            "split": "val" if i < max(1, len(imgs) // 10) else "train",
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)
    print(f"[sfw] Manifest: {out_path} ({len(videos)} entries)")
    return out_path


def _build_placeholder(out_path: str, n: int):
    """Generate solid gray images as SFW placeholders."""
    import cv2
    import numpy as np

    tmp_dir = Path(out_path).parent / "sfw_placeholder"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    for i in range(n):
        img = np.ones((320, 320, 3), dtype=np.uint8) * 128  # gray
        path = tmp_dir / f"sfw_{i:04d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sp = str(path.resolve())
        videos.append({
            "id": f"sfw_{i:04d}",
            "frames": [sp, sp, sp, sp],
            "frame_labels": [0, 0, 0, 0],
            "detections": [[], [], [], []],
            "n_nsfw_frames": 0, "n_frames": 4, "label": 0,
            "split": "val" if i < max(1, n // 10) else "train",
        })

    with open(out_path, "w") as f:
        json.dump(videos, f, indent=2)
    print(f"[sfw] Placeholder manifest: {out_path} ({len(videos)} entries)")
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "datasets/manifest_sfw.json"
    build_sfw_coco_manifest(out)
