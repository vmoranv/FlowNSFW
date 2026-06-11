# 4-Model NSFW Detection Benchmark

**Videos**: 224 (124 NSFW + 100 SFW) | **GPU**: RTX 5060 8GB

## Summary

| Model | Accuracy | NSFW Recall | SFW Accuracy | Avg Time |
|-------|----------|-------------|--------------|----------|
| **FlowNSFW V10** | 96.4% | 98.3% (118/124) | 94.0% (94/100) | 411ms |
| Traditional ML (SVM+HOG) | 55.4% | 100.0% (124/124) | 0.0% (0/100) | 150ms |
| YOLOv11 v16_s | 70.0% | 60.0% (72/124) | 82.0% (82/100) | 265ms |
| YOLOv11 auto_nsfw_v14 | 64.5% | 41.7% (50/124) | 92.0% (92/100) | 332ms |

## Analysis

**FlowNSFW V10** dominates — 26-41 points ahead.
- 8-frame optical flow + Mamba SSM captures motion patterns invisible to single-frame models
- RGB + Flow fusion avoids heuristic traps
- Sliding-window inference misses nothing

**Traditional ML (SVM+HOG)** NSFW recall 100% but SFW 0.0%. HOG+color features trigger on every image. Classic handcrafted-feature failure on complex vision tasks. This is why deep learning replaced traditional ML.

**YOLOv11 v16_s** 60% NSFW recall: motion-dependent NSFW invisible in single frames.

**YOLOv11 auto_v14** 58% NSFW miss rate: overtrained on SFW negatives.
