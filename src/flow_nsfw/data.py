"""Video clip dataset for FlowNSFW.

Reads video frames + YOLO pseudo-labels, yields (T, clip_len) windows.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .pseudo_labeler import _decode_image


def _read_img(path: Path) -> np.ndarray:
    """Read image as RGB uint8, handling AVIF via ffmpeg fallback."""
    return _decode_image(path)


class VideoClipDataset(Dataset):
    """Yields temporal clips from video frame directories.

    Each sample is a fixed-length clip with:
      - frames: (T, 3, H, W) in [0,1]
      - labels: (T,)  int  0=SFW, 1=NSFW  (per-frame from pseudo-label)
      - video_label: int  0=SFW, 1=NSFW  (video-level, from source)

    Uses a manifest file: JSON list of video entries.
    """

    def __init__(
        self,
        manifest: str | Path,
        clip_len: int = 4,
        resolution: tuple[int, int] | list[tuple[int, int]] = (320, 320),
        split: str = "train",
        seed: int = 42,
        frame_stride: int = 2,  # sample every Nth frame
        multi_scale: bool = False,  # Enable random multi-scale per sample
    ):
        super().__init__()
        self.clip_len = clip_len
        # Multi-scale support
        if isinstance(resolution, list):
            self.resolutions = resolution
            self.multi_scale = True
        else:
            self.resolutions = [resolution]
            self.multi_scale = multi_scale
        self.h, self.w = self.resolutions[0] if not multi_scale else (320, 320)
        self.split = split
        self.frame_stride = frame_stride
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)

        self.videos: list[dict] = []
        for v in data:
            if v.get("split", "train") != split:
                continue
            # Allow short videos — __getitem__ repeats frames to fill clip_len
            if len(v["frames"]) < 1:
                continue
            self.videos.append(v)

        if not self.videos:
            raise RuntimeError(f"No videos for split={split} with clip_len={clip_len}")
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        v = self.videos[idx]
        n = len(v["frames"])
        T = self.clip_len
        stride = self.frame_stride
        needed = T * stride

        # Multi-scale: randomly pick resolution for this sample
        if self.multi_scale and self.split == "train":
            h, w = self.rng.choice(self.resolutions)
        else:
            h, w = self.h, self.w

        # If video is shorter than clip, repeat frames
        if n < needed:
            repeat = (needed + n - 1) // n
            frame_paths = v["frames"] * repeat
            frame_labels_list = v.get("frame_labels", [0] * n) * repeat
        else:
            frame_paths = v["frames"]
            frame_labels_list = v.get("frame_labels", [0] * n)

        nn = len(frame_paths)
        max_start = max(0, nn - needed)
        if self.split == "train":
            start = self.rng.randint(0, max_start) if max_start > 0 else 0
        else:
            start = 0

        # Load frames
        frames = np.zeros((T, h, w, 3), dtype=np.uint8)
        labels = np.zeros(T, dtype=np.int64)
        detections_list = v.get("detections", [])
        boxes_per_frame = []  # List of per-frame box tensors

        for i in range(T):
            path = frame_paths[start + i * stride]
            img = _read_img(Path(path))
            if img is None:
                # Corrupted frame, use black placeholder
                img = np.zeros((h, w, 3), dtype=np.uint8)
                orig_h, orig_w = h, w
            else:
                orig_h, orig_w = img.shape[:2]
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            frames[i] = img
            if i < len(frame_labels_list):
                labels[i] = frame_labels_list[start + i * stride]

            # Convert YOLO detections to normalized cxcywh
            det_idx = start + i * stride
            if det_idx < len(detections_list) and detections_list[det_idx]:
                boxes = []
                for det in detections_list[det_idx]:
                    x1, y1, x2, y2 = det["xyxy"]
                    # Normalize to [0,1] and convert to cxcywh
                    cx = ((x1 + x2) / 2) / orig_w
                    cy = ((y1 + y2) / 2) / orig_h
                    bw = (x2 - x1) / orig_w
                    bh = (y2 - y1) / orig_h
                    cls = det.get("cls", 0)
                    boxes.append([cx, cy, bw, bh, cls])
                boxes_per_frame.append(torch.tensor(boxes, dtype=torch.float32))
            else:
                boxes_per_frame.append(torch.zeros(0, 5, dtype=torch.float32))  # Empty

        # To CHW tensor [0,1]
        f_t = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

        return {
            "frames": f_t,                          # (T, 3, H, W)
            "frame_labels": torch.from_numpy(labels),  # (T,)
            "video_label": v.get("label", 1),       # int
            "video_id": v.get("id", str(idx)),
            "boxes": boxes_per_frame,               # List[Tensor] of shape (n_boxes, 5)
        }
