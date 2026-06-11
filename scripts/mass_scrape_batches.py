"""Mass hanime scraper — scrape ALL 301 video pages for MP4 URLs.

Claude runs this page-by-page via jshook:
  1. Navigate to hanime1.me/watch?v=XXXX
  2. page_evaluate: extract source[src] URLs
  3. Save to scraped_urls.jsonl
  4. Repeat with 800ms delay per video

Each batch of 5 takes ~8s. 301 videos = ~60 batches = ~8 min.
"""

import json
from pathlib import Path

# ALL 301 hanime video IDs from our queue
QUEUE = json.load(open("D:/cumhub/flow-nsfw/datasets/hanime_download_queue.json"))

# Flatten: each entry is (domain, video_id)
ALL_VIDS = []
for domain, ids in QUEUE["queue"].items():
    for vid in ids:
        ALL_VIDS.append((domain, vid))

# Split into batches of 5
BATCH_SIZE = 5
for i in range(0, len(ALL_VIDS), BATCH_SIZE):
    batch = ALL_VIDS[i:i + BATCH_SIZE]
    ids_json = json.dumps([vid for _, vid in batch])
    print(f"BATCH|{i//BATCH_SIZE}|{ids_json}")

print(f"\nTotal: {len(ALL_VIDS)} videos in {len(ALL_VIDS)//BATCH_SIZE + 1} batches")
