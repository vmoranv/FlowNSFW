"""Hanime bulk downloader — browser-driven via jshook MCP.

Phase 1 — Collect video IDs across genres/pages (run in browser console).
Phase 2 — For each video, open page → extract mp4 → download via page_evaluate fetch.

Usage: python scripts/hanime_bulk.py --genres "裏番,3DCG,2.5D,MMD" --pages 3 --max-videos 100
"""

import json, sys, time, subprocess, hashlib
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path("D:/cumhub/anti-nsfw-yolo")
RAW = PROJECT / "data-collector" / "raw" / "hanime"
EXTRACTED = PROJECT / "data-collector" / "extracted" / "hanime"
MANIFEST = PROJECT / "data-collector" / "manifests" / "crawl" / "hanime_bulk.jsonl"
SEEN_IDS = set()

# Key: mcp__jshook tools are available when script is called from Claude context
# We write commands to stdout for Claude to execute via jshook
# Then read results from a temp file

GENRES = {
    "裏番": "anime2d",
    "新番預告": "anime2d",
    "泡麵番": "anime2d",
    "Motion Anime": "anime2d",
    "2D動畫": "anime2d",
    "3DCG": "render3d",
    "2.5D": "semi2_5d",
    "AI生成": "semi2_5d",
    "MMD": "render3d",
}


def main():
    """Generate the JS script to run in hanime browser tab."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--genres", default="裏番,3DCG,2.5D,MMD,Motion Anime,2D動畫")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--output", default="D:/cumhub/flow-nsfw/datasets/hanime_video_ids.json")
    args = ap.parse_args()

    genres = [g.strip() for g in args.genres.split(",")]

    # Phase 1: collect video IDs script
    js_collect = """
(async () => {
    const GENRES = %s;
    const PAGES = %d;
    const allVideos = [];
    const seen = new Set();

    for (const genre of GENRES) {
        for (let pg = 1; pg <= PAGES; pg++) {
            const url = `https://hanime1.me/search?genre=${encodeURIComponent(genre)}&sort=views&page=${pg}`;
            console.log(`[collect] ${genre} p${pg}`);
            try {
                const resp = await fetch(url, { credentials: 'include' });
                const html = await resp.text();
                const ids = [...html.matchAll(/watch\\?v=(\\d+)/g)].map(m => m[1]);
                const unique = [...new Set(ids)];
                let added = 0;
                for (const id of unique) {
                    if (!seen.has(id)) {
                        seen.add(id);
                        allVideos.push({ id, genre, url: `https://hanime1.me/watch?v=${id}` });
                        added++;
                    }
                }
                console.log(`  +${added} new (total: ${allVideos.length})`);
            } catch(e) {
                console.log(`  ERROR: ${e.message}`);
            }
            await new Promise(r => setTimeout(r, 2000));
        }
    }
    return JSON.stringify({ total: allVideos.length, videos: allVideos });
})()
""" % (json.dumps(genres), args.pages)

    print("=== PHASE 1: PASTE THIS IN BROWSER CONSOLE (hanime tab) ===")
    print(js_collect)
    print("=== /PHASE 1 ===")


if __name__ == "__main__":
    main()
