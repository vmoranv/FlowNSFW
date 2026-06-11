# Real SFW Video Collection Guide

## Quick Start (Pexels - Recommended)

### 1. Get Pexels API Key (Free)
1. Visit https://www.pexels.com/api/
2. Sign up for free account
3. Copy your API key

### 2. Run Collection Script
```bash
cd D:/cumhub/flow-nsfw/

# Collect 500 SFW videos with frames
python scripts/collect_sfw_videos.py \
  --source pexels \
  --count 500 \
  --out datasets/sfw_videos/ \
  --pexels-key YOUR_API_KEY_HERE \
  --frames-per-video 60 \
  --fps 2
```

**Expected output:**
- `datasets/sfw_videos/videos/pexels_*.mp4` (500 video files)
- `datasets/sfw_videos/videos/pexels_*_frames/` (60 frames per video)
- `datasets/sfw_videos/sfw_manifest.json` (metadata)

### 3. Merge into Training Manifest
```bash
python scripts/merge_manifests.py \
  --base datasets/manifest_final.json \
  --new datasets/sfw_videos/sfw_manifest.json \
  --out datasets/manifest_v3_multiscale.json \
  --label 0  # SFW
```

### 4. Retrain with Multi-Scale
```bash
D:/cumhub/anti-nsfw-yolo/.venv2/Scripts/python.exe scripts/train.py \
  --manifest datasets/manifest_v3_multiscale.json \
  --epochs 40 \
  --batch-size 2 \
  --clip-len 4 \
  --lr 2e-4 \
  --dim 128 \
  --num-heads 4 \
  --num-temporal-layers 3 \
  --topk-global 64 \
  --multi-scale \
  --resolutions 160 240 320 480 \
  --log-every 20 \
  --ckpt-every 3000 \
  --out runs/flow_nsfw_v3_multiscale \
  --bf16 \
  --device cuda \
  --seed 42
```

---

## Alternative: YouTube-8M Segments

### 1. Download Segment List
```bash
# Get YouTube-8M frame-level annotations
wget http://us.data.yt8m.org/2/frame/train/train*.tfrecord
```

### 2. Extract Video URLs
```python
# Use TensorFlow to decode .tfrecord → video IDs
# Filter by safe categories (sports, nature, cooking, etc.)
```

### 3. Download with yt-dlp
```bash
yt-dlp \
  --format "bestvideo[height<=720]+bestaudio/best[height<=720]" \
  --max-downloads 500 \
  --output "datasets/sfw_videos/yt8m/%(id)s.%(ext)s" \
  --batch-file youtube_ids.txt
```

---

## Alternative: Manual Curation

### Public Domain Sources
1. **Pixabay Videos** - https://pixabay.com/videos/
2. **Videvo** - https://www.videvo.net/
3. **Coverr** - https://coverr.co/
4. **Mixkit** - https://mixkit.co/free-stock-video/

### Download Script
```bash
# Use gallery-dl or yt-dlp
gallery-dl --range 1-100 "https://pixabay.com/videos/search/nature/"
```

---

## Dataset Quality Checklist

✅ **Real motion** — not repeated single frames
✅ **Diverse scenes** — indoor, outdoor, day, night
✅ **Multiple subjects** — people, animals, objects, nature
✅ **Various lighting** — bright, dim, backlit, shadows
✅ **Camera motion** — panning, zooming, handheld, static
✅ **Different durations** — 5s-30s clips
✅ **HD resolution** — at least 720p source

---

## Expected Improvement

After adding 500 real SFW videos:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Val Accuracy @ 480p | 81.2% | **95%+** |
| False Positives | 3/10 | **0/10** |
| SFW Precision | 70% | **100%** |

---

## Troubleshooting

### "Rate limit exceeded" (Pexels)
- Add `time.sleep(2)` between downloads
- Reduce `--count` to 200, run multiple times

### "ffmpeg not found"
```bash
# Windows (via scoop)
scoop install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### "Out of memory during training"
- Reduce `--batch-size` to 1
- Remove 480 from `--resolutions` (only use 160 240 320)
- Lower `--dim` to 96

---

## Next Steps

1. ✅ Collect SFW videos (this guide)
2. Merge manifests
3. Retrain with `--multi-scale`
4. Eval on 480×480 → should see 0 FP
5. Move to Task #4: Activate optical flow
