"""Bulk video scraper + downloader.

Phase 1 — Scrape MP4 URLs (jshook browser driving):
  Hanime: page_evaluate extracts source[src] → vdownload.hembed.com/...
  Iwara:  page_evaluate extracts fileUrl from scripts → xxx.iwara.tv/view?hash=...

Phase 2 — Download (yt-dlp + curl):

Usage for Claude: save scrape results to scraped_urls.jsonl, then:
  .venv2/Scripts/python.exe scripts/bulk_download.py --download --max 50
"""

import json, subprocess, hashlib, time, sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path("D:/cumhub/anti-nsfw-yolo")
RAW = PROJECT / "data-collector" / "raw"
EXTRACTED = PROJECT / "data-collector" / "extracted"
MANIFEST_DIR = PROJECT / "data-collector" / "manifests" / "crawl"
SCRAPED = MANIFEST_DIR / "scraped_urls.jsonl"
RESULTS = MANIFEST_DIR / "download_results.jsonl"

YTDLP = str(PROJECT / ".venv2" / "Scripts" / "yt-dlp.exe")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/148.0.0.0 Safari/537.36"


def extract_frames(domain: str, vid_hash: str, video_path: Path):
    d = EXTRACTED / domain / vid_hash
    d.mkdir(parents=True, exist_ok=True)
    if len(list(d.glob("f*.jpg"))) > 0:
        return
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", "fps=1/5", "-vframes", "60", "-q:v", "2", "-y",
        str(d / "f%04d.jpg"),
    ], capture_output=True, timeout=120)


def download_one(entry: dict) -> dict:
    url = entry["url"]
    domain = entry["domain"]
    vid = entry["video_id"]
    vid_hash = hashlib.sha256(vid.encode()).hexdigest()[:12]
    out_path = RAW / domain / f"{vid_hash}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 50_000:
        extract_frames(domain, vid_hash, out_path)
        n = len(list((EXTRACTED / domain / vid_hash).glob("f*.jpg")))
        return {**entry, "status": "cached", "vid_hash": vid_hash, "size_mb": round(out_path.stat().st_size/1e6,1), "n_frames": n}

    referer = entry.get("referer", "https://hanime1.me/" if "hanime" in entry.get("source","") else "https://www.iwara.tv/")

    t0 = time.time()
    r = subprocess.run([
        YTDLP,
        "--referer", referer,
        "-o", str(out_path),
        "--no-playlist",
        "--socket-timeout", "30", "--retries", "2",
        url,
    ], capture_output=True, text=True, timeout=120 if "iwara" in url else 180)

    ok = out_path.exists() and out_path.stat().st_size > 50_000
    if not ok:
        out_path.unlink(missing_ok=True)
        return {**entry, "status": "fail", "reason": r.stderr[-200:] if r.stderr else "unknown"}

    extract_frames(domain, vid_hash, out_path)
    n_frames = len(list((EXTRACTED / domain / vid_hash).glob("f*.jpg")))
    out_path.unlink(missing_ok=True)  # save space

    return {**entry, "status": "ok", "vid_hash": vid_hash,
            "size_mb": round(out_path.stat().st_size/1e6, 1) if out_path.exists() else 0,
            "n_frames": n_frames, "elapsed_s": round(time.time()-t0, 1),
            "downloaded_at": datetime.now(timezone.utc).isoformat()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="Run downloads from scraped URLs")
    ap.add_argument("--max", type=int, default=50, help="Max videos to download")
    ap.add_argument("--domain", default="all", help="Filter by domain")
    args = ap.parse_args()

    if not args.download:
        print("Use --download to start downloading from scraped_urls.jsonl")
        print("First scrape URLs in browser:")
        print("  Hanime JS: document.querySelectorAll('source[src]') → vdownload.hembed.com/...")
        print("  Iwara JS:  scripts → fileUrl → xxx.iwara.tv/view?hash=...")
        print(f"  Save to: {SCRAPED}")
        return 1

    if not SCRAPED.exists():
        print(f"No scraped URLs yet: {SCRAPED}")
        return 1

    # Load done
    done = set()
    if RESULTS.exists():
        with open(RESULTS) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["video_id"])
                    except: pass

    entries = []
    with open(SCRAPED) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["video_id"] not in done:
                    if args.domain == "all" or e["domain"] == args.domain:
                        entries.append(e)

    entries = entries[:args.max]
    print(f"Downloading {len(entries)} videos ({args.max} max, domain={args.domain})")

    ok = fail = 0
    for i, entry in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] {entry['domain']}/{entry['video_id']}", end=" ", flush=True)
        r = download_one(entry)
        if r["status"] in ("ok", "cached"):
            ok += 1
            print(f"OK ({r.get('n_frames',0)} frames, {r.get('size_mb',0):.0f}MB)")
        else:
            fail += 1
            print(f"FAIL")

        with open(RESULTS, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone: ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
