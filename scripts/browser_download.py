"""Browser-driven batch downloader — scrapes video URLs via jshook page_evaluate, downloads via curl.

Phase 1: Browser scrape — extract direct .mp4 URLs from video pages.
Phase 2: Shell download — curl the URLs.

Can be split across multiple sessions: scrape first, download later.
"""

import asyncio, json, subprocess, hashlib, time, sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path("D:/cumhub/anti-nsfw-yolo")
RAW = PROJECT / "data-collector" / "raw"
EXTRACTED = PROJECT / "data-collector" / "extracted"
MANIFEST_DIR = PROJECT / "data-collector" / "manifests" / "crawl"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/148.0.0.0 Safari/537.36"


def extract_frames(domain: str, vid_hash: str, video_path: Path, interval: int = 5, max_frames: int = 60) -> int:
    """Extract JPG frames via ffmpeg."""
    frame_dir = EXTRACTED / domain / vid_hash
    frame_dir.mkdir(parents=True, exist_ok=True)
    n_existing = len(list(frame_dir.glob("f*.jpg")))
    if n_existing > 0:
        return n_existing

    pattern = str(frame_dir / "f%04d.jpg")
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"fps=1/{interval}", "-vframes", str(max_frames),
        "-q:v", "2", "-y", pattern,
    ], capture_output=True, timeout=120)
    return len(list(frame_dir.glob("f*.jpg")))


def download_url(url: str, domain: str, vid_hash: str, referer: str = "") -> dict:
    """Download a single video URL via curl."""
    vid_dir = RAW / domain
    vid_dir.mkdir(parents=True, exist_ok=True)
    out_path = vid_dir / f"{vid_hash}.mp4"

    if out_path.exists() and out_path.stat().st_size > 100_000:
        n_frames = extract_frames(domain, vid_hash, out_path)
        return {"status": "cached", "video_id": vid_hash, "domain": domain,
                "n_frames": n_frames, "size_mb": round(out_path.stat().st_size / 1e6, 1)}

    cmd = ["curl", "-s", "-L", "-o", str(out_path),
           "--max-time", "120", "--retry", "2",
           "-H", f"User-Agent: {UA}",
           "-H", f"Referer: {referer}",
           url]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=150)

    if not out_path.exists() or out_path.stat().st_size < 50_000:
        out_path.unlink(missing_ok=True)
        return {"status": "fail", "video_id": vid_hash, "domain": domain,
                "reason": f"download failed ({out_path.stat().st_size if out_path.exists() else 0} bytes)"}

    size_mb = out_path.stat().st_size / 1e6
    n_frames = extract_frames(domain, vid_hash, out_path)
    # Delete video to save space
    out_path.unlink(missing_ok=True)

    return {"status": "ok", "video_id": vid_hash, "domain": domain,
            "size_mb": round(size_mb, 1), "n_frames": n_frames,
            "elapsed_s": round(time.time() - t0, 1),
            "downloaded_at": datetime.now(timezone.utc).isoformat()}


# =========================================================================
# The jshook scrape scripts — run in browser to collect MP4 URLs
# =========================================================================

IWARA_EXTRACT_JS = """
(() => {
  const sources = [];
  document.querySelectorAll('video source, video').forEach(el => {
    const src = el.src || el.getAttribute('src');
    if (src && src.includes('.mp4')) sources.push(src);
  });
  document.querySelectorAll('a[href*=".mp4"]').forEach(a => sources.push(a.href));
  [...document.querySelectorAll('script')].forEach(s => {
    const m = s.textContent.match(/fileUrl["']?\s*[:=]\s*["']([^"']+\\.mp4[^"']*)["']/g);
    if (m) m.forEach(x => {
      const u = x.match(/["']([^"']+\\.mp4[^"']*)["']/);
      if (u) sources.push(u[1]);
    });
  });
  return JSON.stringify({ sources: [...new Set(sources)].filter(s => s.includes('/view?hash=')) });
})()
"""

HANIME_EXTRACT_JS = """
(() => {
  const sources = [];
  document.querySelectorAll('video source, video').forEach(el => {
    const src = el.src || el.getAttribute('src');
    if (src && (src.includes('.mp4') || src.includes('.m3u8'))) sources.push(src);
  });
  document.querySelectorAll('a[href*=".mp4"], a[href*="vdownload"]').forEach(a => sources.push(a.href));
  // Hem bed URLs in scripts
  [...document.querySelectorAll('script')].forEach(s => {
    const m = s.textContent.match(/https?:\\/\\/vdownload[^"'<>\\s]+\\.mp4[^"'<>\\s]*/g);
    if (m) sources.push(...m);
  });
  return JSON.stringify({ sources: [...new Set(sources)] });
})()
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", help="anime2d, render3d, semi2_5d, iwara, or all")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--scrape-only", action="store_true", help="Only output JS commands, don't download")
    ap.add_argument("--download-only", action="store_true", help="Download from existing URL manifest")
    args = ap.parse_args()

    queue = json.load(open("D:/cumhub/flow-nsfw/datasets/hanime_download_queue.json"))

    if args.scrape_only:
        print("=== IWARA SCRAPE JS (paste in iwara tab console) ===")
        print(IWARA_EXTRACT_JS)
        print("=== HANIME SCRAPE JS (paste in hanime tab console) ===")
        print(HANIME_EXTRACT_JS)
        sys.exit(0)

    if args.download_only:
        # Download from saved URL manifest
        manifest_path = MANIFEST_DIR / f"scraped_urls_{args.domain}.json"
        if not manifest_path.exists():
            print(f"No URL manifest found: {manifest_path}")
            print("Run --scrape-only first to get JS commands, scrape in browser, then save results")
            sys.exit(1)
        urls = json.loads(manifest_path.read_text())
        ok = fail = 0
        for i, entry in enumerate(urls[:args.max]):
            url = entry["url"]
            domain = entry["domain"]
            vid_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
            referer = entry.get("referer", "")
            print(f"[{i+1}/{min(args.max, len(urls))}] {domain}/{vid_hash}", end=" ", flush=True)
            r = download_url(url, domain, vid_hash, referer)
            if r["status"] in ("ok", "cached"):
                ok += 1
                print(f"OK ({r['n_frames']} frames)")
            else:
                fail += 1
                print(f"FAIL: {r.get('reason', '?')}")
        print(f"\nDone: ok={ok} fail={fail}")
        sys.exit(0)

    # Default: print instructions
    print("=== Browser Scrape + Download Instructions ===")
    print(f"1. For IWARA: navigate to each video page, run:")
    print(f"   page_evaluate: {IWARA_EXTRACT_JS[:200]}...")
    print(f"2. For HANIME: navigate to each video page, run:")
    print(f"   page_evaluate: {HANIME_EXTRACT_JS[:200]}...")
    print(f"3. Save results to D:/cumhub/flow-nsfw/datasets/scraped_urls.json")
    print(f"4. Run: python scripts/batch_download.py --download-only")
