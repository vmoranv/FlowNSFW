"""Collect real SFW video clips from public datasets.

Data sources:
1. Pexels API (free, CC0 license)
2. YouTube-8M (subset download via URLs)
3. Open Images Extended (video version)

Usage:
    python scripts/collect_sfw_videos.py --source pexels --count 500 --out datasets/sfw_videos/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Iterator

import requests


# --- Pexels API ---
def collect_pexels(
    api_key: str,
    count: int,
    out_dir: Path,
    categories: list[str] | None = None,
) -> list[dict]:
    """Download video clips from Pexels API.

    Args:
        api_key: Pexels API key (get from https://www.pexels.com/api/)
        count: Target number of videos
        out_dir: Output directory
        categories: Search queries (e.g., ["nature", "city", "people"])

    Returns:
        List of downloaded video metadata
    """
    if categories is None:
        categories = [
            "nature landscape",
            "city traffic",
            "people walking",
            "cooking food",
            "sports workout",
            "office work",
            "family home",
            "children playing",
            "animals pets",
            "travel tourism",
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": api_key}
    base_url = "https://api.pexels.com/videos/search"

    collected = []
    per_category = (count + len(categories) - 1) // len(categories)

    for cat in categories:
        print(f"[pexels] Searching: {cat}")
        params = {
            "query": cat,
            "per_page": min(80, per_category * 2),  # Get more, filter later
            "orientation": "landscape",
            "size": "medium",  # 720p
        }

        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[pexels] Error: {e}")
            continue

        videos = data.get("videos", [])
        print(f"[pexels] Found {len(videos)} videos")

        for vid in videos[:per_category]:
            if len(collected) >= count:
                break

            video_id = vid["id"]
            # Get HD file
            files = vid.get("video_files", [])
            hd_file = None
            for f in files:
                if f.get("quality") == "hd" and f.get("width", 0) >= 1280:
                    hd_file = f
                    break
            if not hd_file and files:
                hd_file = files[0]  # Fallback to first

            if not hd_file:
                continue

            download_url = hd_file["link"]
            duration = vid.get("duration", 0)

            # Download
            out_path = out_dir / f"pexels_{video_id}.mp4"
            if out_path.exists():
                print(f"[pexels] Skip existing: {out_path.name}")
                collected.append({
                    "id": f"pexels_{video_id}",
                    "source": "pexels",
                    "path": str(out_path),
                    "duration": duration,
                    "category": cat,
                })
                continue

            try:
                print(f"[pexels] Downloading: {out_path.name} ({duration}s)")
                r = requests.get(download_url, stream=True, timeout=60)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)

                collected.append({
                    "id": f"pexels_{video_id}",
                    "source": "pexels",
                    "path": str(out_path),
                    "duration": duration,
                    "category": cat,
                })
                time.sleep(1)  # Rate limit
            except Exception as e:
                print(f"[pexels] Download failed: {e}")
                if out_path.exists():
                    out_path.unlink()

        if len(collected) >= count:
            break

    return collected


def extract_frames_from_videos(
    video_list: list[dict],
    frames_per_video: int = 60,
    fps: int = 2,
) -> list[dict]:
    """Extract frames from collected videos using ffmpeg.

    Args:
        video_list: List of video metadata dicts
        frames_per_video: Max frames to extract per video
        fps: Frame extraction rate

    Returns:
        Updated video list with frame paths
    """
    results = []
    for vid in video_list:
        video_path = Path(vid["path"])
        if not video_path.exists():
            continue

        out_frame_dir = video_path.parent / f"{video_path.stem}_frames"
        out_frame_dir.mkdir(exist_ok=True)

        # Extract frames
        duration = vid.get("duration", 30)
        max_frames = min(frames_per_video, int(duration * fps))

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            "-q:v", "2",  # High quality
            str(out_frame_dir / "f%04d.jpg"),
            "-y",
        ]

        try:
            print(f"[ffmpeg] Extracting {max_frames} frames from {video_path.name}")
            subprocess.run(cmd, check=True, capture_output=True)

            # List extracted frames
            frame_paths = sorted(out_frame_dir.glob("f*.jpg"))
            if frame_paths:
                vid["frames"] = [str(p) for p in frame_paths]
                vid["n_frames"] = len(frame_paths)
                results.append(vid)
                print(f"[ffmpeg] Extracted {len(frame_paths)} frames")
        except subprocess.CalledProcessError as e:
            print(f"[ffmpeg] Error: {e.stderr.decode()[:200]}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pexels", "youtube8m"], default="pexels")
    ap.add_argument("--count", type=int, default=500, help="Target video count")
    ap.add_argument("--out", default="datasets/sfw_videos/", help="Output directory")
    ap.add_argument("--pexels-key", default="", help="Pexels API key")
    ap.add_argument("--frames-per-video", type=int, default=60)
    ap.add_argument("--fps", type=int, default=2, help="Frame extraction rate")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.source == "pexels":
        if not args.pexels_key:
            print("[ERROR] --pexels-key required. Get from: https://www.pexels.com/api/")
            print("        Sign up (free) and copy your API key.")
            return

        videos = collect_pexels(args.pexels_key, args.count, out_dir / "videos")
        print(f"\n[collected] {len(videos)} videos")

        # Extract frames
        videos_with_frames = extract_frames_from_videos(
            videos,
            frames_per_video=args.frames_per_video,
            fps=args.fps,
        )

        # Save manifest
        manifest_path = out_dir / "sfw_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(videos_with_frames, f, indent=2)

        print(f"\n[done] {len(videos_with_frames)} videos with frames")
        print(f"[done] Manifest: {manifest_path}")
        print(f"[done] Total frames: {sum(v['n_frames'] for v in videos_with_frames)}")

    elif args.source == "youtube8m":
        print("[TODO] YouTube-8M download not implemented yet")
        print("       Use yt-dlp to download from YouTube-8M segment URLs")


if __name__ == "__main__":
    main()
