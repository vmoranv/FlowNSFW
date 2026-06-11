"""Evaluate FlowNSFW at multiple resolutions.

Usage:
    python scripts/eval_multi_res.py --ckpt runs/flow_nsfw_v3_final/final.pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset


def eval_at_resolution(model, manifest, resolution, device):
    """Evaluate at a single resolution."""
    ds = VideoClipDataset(manifest, clip_len=4, resolution=resolution, split="val")

    yt, yp = [], []
    for i in range(len(ds)):
        s = ds[i]
        l = s["video_label"]
        if hasattr(l, "item"):
            l = l.item()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(s["frames"].unsqueeze(0).to(device))

        yt.append(l)
        yp.append(o["video_cls"].argmax(-1).item())

    # Confusion matrix
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)

    acc = (tp + tn) / len(yt) * 100
    rec = tp / max(1, tp + fn) * 100
    prec = tp / max(1, tp + fp) * 100

    return {
        "resolution": resolution,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Model checkpoint")
    ap.add_argument("--manifest", default="datasets/manifest_v3_with_real_sfw.json")
    ap.add_argument("--resolutions", nargs="+", type=int, default=[160, 240, 320, 480, 640])
    ap.add_argument("--temporal-backend", choices=["attention", "mamba", "hybrid"],
                    default="attention")
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--ssm-expand", type=int, default=2)
    ap.add_argument("--sparse-detect", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load model
    ck = torch.load(args.ckpt, map_location=device)
    model = FlowNSFW(
        dim=128, num_heads=4, num_temporal_layers=3, topk_global=64,
        temporal_backend=args.temporal_backend,
        d_state=args.d_state, ssm_expand=args.ssm_expand,
        sparse_detect=args.sparse_detect,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params:.2f}M params, step={ck.get('step', '?')}\n")

    # Evaluate at all resolutions
    results = []
    for res in args.resolutions:
        print(f"Evaluating @ {res}×{res}...", end=" ", flush=True)
        r = eval_at_resolution(model, ROOT / args.manifest, (res, res), device)
        results.append(r)
        print(f"Acc={r['accuracy']:.1f}% Prec={r['precision']:.1f}% Rec={r['recall']:.1f}%")

    # Summary table
    print("\n" + "="*70)
    print(f"{'Resolution':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'FP/FN'}")
    print("="*70)
    for r in results:
        res_str = f"{r['resolution'][0]}×{r['resolution'][1]}"
        print(f"{res_str:<12} {r['accuracy']:>6.1f}%   {r['precision']:>6.1f}%   "
              f"{r['recall']:>6.1f}%   {r['fp']}/{r['fn']}")
    print("="*70)

    # Check if 480p target met
    r480 = next((r for r in results if r['resolution'][0] == 480), None)
    if r480:
        if r480['accuracy'] >= 95.0 and r480['fp'] == 0:
            print("\n🎉 TARGET MET: 480p ≥95% accuracy, 0 false positives!")
        else:
            print(f"\n⚠️  480p: {r480['accuracy']:.1f}% (target: 95%+), FP={r480['fp']} (target: 0)")


if __name__ == "__main__":
    main()
