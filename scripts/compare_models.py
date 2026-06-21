#!/usr/bin/env python3
"""对比原始模型 vs 优化生产模型的评测结果."""

import json, sys
from pathlib import Path

def compare():
    baseline = Path("runs/eval_baseline.json")
    prod = Path("runs/eval_prod_final.json")

    if not baseline.exists():
        print("ERROR: baseline eval not found. Run first:")
        print("  python3 scripts/eval_production.py --ckpt final.pt --manifest datasets/manifest_v4_clean_wsl.json --mode mamba2_baseline --resolution 320 --output runs/eval_baseline.json")
        return

    if not prod.exists():
        print("ERROR: prod eval not found. Run first:")
        print("  python3 scripts/eval_production.py --ckpt runs/prod_final/final.pt --manifest datasets/manifest_v4_clean_wsl.json --mode mamba2_full --resolution 192 --output runs/eval_prod_final.json")
        return

    b = json.loads(baseline.read_text())
    p = json.loads(prod.read_text())

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} {'Baseline (final.pt)':<20} {'Optimized (prod)':<20} {'Delta':<10}")
    print("-"*60)
    print(f"{'Accuracy':<25} {b['accuracy']:<20.1f}% {p['accuracy']:<20.1f}% {p['accuracy']-b['accuracy']:+.1f}%")
    print(f"{'NSFW Recall':<25} {b['nsfw_recall']:<20.1f}% {p['nsfw_recall']:<20.1f}% {p['nsfw_recall']-b['nsfw_recall']:+.1f}%")
    print(f"{'SFW Accuracy':<25} {b['sfw_accuracy']:<20.1f}% {p['sfw_accuracy']:<20.1f}% {p['sfw_accuracy']-b['sfw_accuracy']:+.1f}%")
    print(f"{'Resolution':<25} {b['resolution']:<20} {p['resolution']:<20}")
    print(f"{'Params':<25} {b['params_m']:<20.2f}M {p['params_m']:<20.2f}M")
    print(f"{'Time':<25} {b['elapsed_s']:<20.1f}s {p['elapsed_s']:<20.1f}s")
    print("="*60)

    # Best/worst analysis
    b_fails = [r for r in b['results'] if not r['ok']]
    p_fails = [r for r in p['results'] if not r['ok']]
    print(f"\nBaseline failures: {len(b_fails)}")
    for r in b_fails:
        print(f"  {r['video_id'][:30]}: pred={r['verdict']} gt={r['gt']} conf={r['max_conf']:.3f}")
    print(f"\nOptimized failures: {len(p_fails)}")
    for r in p_fails:
        print(f"  {r['video_id'][:30]}: pred={r['verdict']} gt={r['gt']} conf={r['max_conf']:.3f}")

if __name__ == "__main__":
    compare()
