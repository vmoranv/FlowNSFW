"""Merge multiple manifest files into one.

Usage:
    python scripts/merge_manifests.py \
        --base datasets/manifest_final.json \
        --new datasets/sfw_videos/sfw_manifest.json \
        --out datasets/manifest_v3_multiscale.json \
        --label 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def merge_manifests(
    base_path: Path,
    new_path: Path,
    out_path: Path,
    label: int = 0,
    split_ratio: float = 0.1,
    seed: int = 42,
):
    """Merge new videos into existing manifest.

    Args:
        base_path: Existing manifest (e.g., NSFW videos)
        new_path: New videos to add (e.g., SFW videos from Pexels)
        out_path: Output merged manifest
        label: Label for new videos (0=SFW, 1=NSFW)
        split_ratio: Fraction of new videos for validation
        seed: Random seed for train/val split
    """
    # Load existing
    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)

    print(f"[base] {len(base)} videos")

    # Load new
    with open(new_path, encoding="utf-8") as f:
        new_videos = json.load(f)

    print(f"[new] {len(new_videos)} videos")

    # Convert new videos to manifest format
    rng = random.Random(seed)
    val_count = int(len(new_videos) * split_ratio)
    val_indices = set(rng.sample(range(len(new_videos)), val_count))

    converted = []
    for i, v in enumerate(new_videos):
        split = "val" if i in val_indices else "train"
        converted.append({
            "id": v.get("id", f"new_{i}"),
            "source": v.get("source", "unknown"),
            "domain": v.get("category", "sfw"),
            "label": label,
            "frames": v.get("frames", []),
            "n_frames": v.get("n_frames", len(v.get("frames", []))),
            "split": split,
        })

    # Merge
    merged = base + converted
    print(f"[merged] {len(merged)} total videos")

    # Stats
    train_count = sum(1 for v in merged if v.get("split") == "train")
    val_count = sum(1 for v in merged if v.get("split") == "val")
    nsfw_count = sum(1 for v in merged if v.get("label") == 1)
    sfw_count = sum(1 for v in merged if v.get("label") == 0)

    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  NSFW:  {nsfw_count}")
    print(f"  SFW:   {sfw_count}")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base manifest")
    ap.add_argument("--new", required=True, help="New videos to add")
    ap.add_argument("--out", required=True, help="Output merged manifest")
    ap.add_argument("--label", type=int, default=0, help="Label for new videos (0=SFW, 1=NSFW)")
    ap.add_argument("--split-ratio", type=float, default=0.1, help="Val split for new videos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    merge_manifests(
        Path(args.base),
        Path(args.new),
        Path(args.out),
        label=args.label,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
