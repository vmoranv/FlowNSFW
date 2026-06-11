"""Overnight batch video downloader.

Uses yt-dlp with Chrome browser cookies (no curl/CF issues).
Reads URL lists from data-collector/manifests/crawl/ and downloads sequentially.

Usage:
    .venv2/Scripts/python.exe scripts/batch_download.py --domain anime2d --max 30
    .venv2/Scripts/python.exe scripts/batch_download.py --all --max 200
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("D:/cumhub/anti-nsfw-yolo")
RAW = PROJECT / "data-collector" / "raw"
EXTRACTED = PROJECT / "data-collector" / "extracted"
MANIFEST_DIR = PROJECT / "data-collector" / "manifests" / "crawl"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# yt-dlp reads cookies directly from Chrome
YTDLP_BASE = [
    "yt-dlp",
    "--cookies-from-browser", "chrome",
    "--no-check-certificate",
    "--socket-timeout", "30",
    "--retries", "3",
    "--fragment-retries", "3",
    "--no-playlist",
    "--no-warnings",
]

FFMPEG_BASE = ["ffmpeg", "-v", "error"]


def download_and_extract(url: str, domain: str, keep_video: bool = False) -> dict:
    """Download a video and extract frames. Returns stats dict."""
    vid_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    vid_dir = RAW / domain
    vid_dir.mkdir(parents=True, exist_ok=True)
    output_tpl = str(vid_dir / f"{vid_hash}.%(ext)s")

    # Step 1: yt-dlp download
    t0 = time.time()
    result = subprocess.run(
        YTDLP_BASE + ["-o", output_tpl, url],
        capture_output=True, text=True, timeout=300,
        cwd=str(PROJECT),
    )

    # Find downloaded file
    downloaded = None
    for ext in [".mp4", ".webm", ".mkv"]:
        candidate = vid_dir / f"{vid_hash}{ext}"
        if candidate.exists() and candidate.stat().st_size > 50_000:
            downloaded = candidate
            break

    if not downloaded:
        return {"status": "fail", "reason": "download failed or too small",
                "url": url, "stderr": result.stderr[-300:]}

    size_mb = downloaded.stat().st_size / (1024 * 1024)

    # Step 2: FFmpeg extract frames
    frame_dir = EXTRACTED / domain / vid_hash
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames at 5s intervals, max 60 frames per video
    frame_pattern = str(frame_dir / "f%04d.jpg")
    ext_result = subprocess.run(
        FFMPEG_BASE + [
            "-i", str(downloaded),
            "-vf", "fps=1/5",
            "-vframes", "60",
            "-q:v", "2",
            "-y",
            frame_pattern,
        ],
        capture_output=True, text=True, timeout=120,
    )

    n_frames = len(list(frame_dir.glob("f*.jpg")))

    if not keep_video:
        downloaded.unlink()

    elapsed = time.time() - t0

    return {
        "status": "ok",
        "video_id": vid_hash,
        "domain": domain,
        "url": url,
        "size_mb": round(size_mb, 1),
        "n_frames": n_frames,
        "elapsed_s": round(elapsed, 1),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def process_url_list(url_file: Path, domain: str, max_count: int,
                     keep_video: bool, manifest_path: Path) -> dict:
    """Process a URL list file. Returns totals."""
    urls = [line.strip() for line in url_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")]

    # Load already-done
    done = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["url"])
                    except Exception:
                        pass

    new_urls = [u for u in urls if u not in done][:max_count]
    if not new_urls:
        print(f"  [{domain}] All {len(urls)} already downloaded")
        return {"ok": 0, "fail": 0, "skip": len(urls)}

    print(f"  [{domain}] {len(new_urls)} new / {len(urls)} total")

    ok, fail = 0, 0
    for i, url in enumerate(new_urls):
        print(f"    [{i+1}/{len(new_urls)}] {url[:80]}...", end=" ", flush=True)
        result = download_and_extract(url, domain, keep_video)
        if result["status"] == "ok":
            ok += 1
            print(f"OK ({result['n_frames']} frames, {result['size_mb']:.1f}MB, {result['elapsed_s']:.0f}s)")
        else:
            fail += 1
            print(f"FAIL: {result.get('reason', 'unknown')}")

        # Append to manifest
        with open(manifest_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if i < len(new_urls) - 1:
            time.sleep(2)  # polite delay

    return {"ok": ok, "fail": fail, "skip": len(done)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="", help="Specific domain to download")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max", type=int, default=30, help="Max videos per domain")
    ap.add_argument("--keep-videos", action="store_true")
    args = ap.parse_args()

    DOMAIN_FILES = {
        "hanime_anime2d": ("hanime_anime2d_urls.txt", "anime2d"),
        "hanime_render3d": ("hanime_render3d_urls.txt", "render3d"),
        "hanime_semi2_5d": ("hanime_semi2_5d_urls.txt", "semi2_5d"),
        "iwara": ("iwara_urls.txt", "iwara"),
    }

    to_process = []
    if args.all:
        to_process = list(DOMAIN_FILES.items())
    elif args.domain:
        matches = [(k, v) for k, v in DOMAIN_FILES.items() if args.domain in k]
        to_process = matches or [(args.domain, (f"{args.domain}_urls.txt", args.domain))]

    if not to_process:
        print("No domains selected. Use --all or --domain <name>")
        return 1

    total = {"ok": 0, "fail": 0, "skip": 0}
    for name, (filename, domain) in to_process:
        url_file = MANIFEST_DIR / filename
        if not url_file.exists():
            print(f"  [{name}] URL file not found: {url_file}")
            continue
        manifest = MANIFEST_DIR / f"batch_{name}_results.jsonl"
        r = process_url_list(url_file, domain, args.max, args.keep_videos, manifest)
        for k in total:
            total[k] += r[k]

    print(f"\nDone: ok={total['ok']} fail={total['fail']} skip={total['skip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
