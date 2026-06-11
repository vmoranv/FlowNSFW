"""Train FlowNSFW — optical-flow NSFW detection.

Usage (must use anti-nsfw-yolo .venv2):
    D:/cumhub/anti-nsfw-yolo/.venv2/Scripts/python.exe scripts/train.py \
        --manifest datasets/manifest.json \
        --epochs 50 --batch 1 --clip-len 4 \
        --bf16 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Auto-detect: if running outside venv, warn
_VENV_PYTHON = Path("D:/cumhub/anti-nsfw-yolo/.venv2/Scripts/python.exe")
if _VENV_PYTHON.exists() and sys.executable != str(_VENV_PYTHON):
    print(f"[WARN] Not using .venv2! Run with: {_VENV_PYTHON} {' '.join(sys.argv)}")
    print("[WARN] Continuing anyway...")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from flow_nsfw import FlowNSFW
from flow_nsfw.data import VideoClipDataset
from flow_nsfw.balanced_sampler import BalancedBatchSampler
from flow_nsfw.losses import (
    LossWeights, detection_loss, video_cls_loss, temporal_box_loss,
)


def collate_simple(batch):
    """Collate for balanced batch — all same resolution, handle boxes as list."""
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "frame_labels": torch.stack([b["frame_labels"] for b in batch]),
        "video_label": torch.tensor([b["video_label"] for b in batch]),
        "video_id": [b["video_id"] for b in batch],
        "boxes": [b["boxes"] for b in batch],
    }


def _cosine_lr(step: int, max_step: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, max_step - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


def _set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON video manifest")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--clip-len", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--out", default="runs/flow_nsfw")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-temporal-layers", type=int, default=3)
    ap.add_argument("--topk-global", type=int, default=64)
    ap.add_argument("--flow-backend", choices=["scratch", "raft"], default="scratch")
    ap.add_argument("--temporal-backend", choices=["attention", "mamba", "hybrid"],
                    default="attention", help="Temporal aggregation backend")
    ap.add_argument("--d-state", type=int, default=16, help="SSM state size (mamba/hybrid)")
    ap.add_argument("--ssm-expand", type=int, default=2, help="SSM expand factor (mamba/hybrid)")
    ap.add_argument("--sparse-detect", action="store_true",
                    help="Enable foreground-gated sparse detection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", default="")
    ap.add_argument("--multi-scale", action="store_true", help="Enable random multi-scale training")
    ap.add_argument("--resolutions", nargs="+", type=int, default=[160, 240, 320, 480],
                    help="Resolutions for multi-scale (default: 160 240 320 480)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    # Fixed resolution + balanced batch sampler for contrastive learning
    if args.clip_len >= 8:
        resolution = (192, 192)  # 8-frame clips need less spatial res to fit VRAM
    else:
        resolution = (256, 256)
    print(f"[flow-nsfw] resolution={resolution}, clip_len={args.clip_len}, balanced_batch=True")

    train_ds = VideoClipDataset(
        manifest=args.manifest, clip_len=args.clip_len,
        resolution=resolution, split="train", seed=args.seed,
    )
    val_ds = VideoClipDataset(
        manifest=args.manifest, clip_len=args.clip_len,
        resolution=(320, 320), split="val", seed=args.seed,
    )
    # Balanced: each batch = 1 NSFW + 1 SFW for contrastive learning
    # Balanced: read labels from manifest (fast) instead of iterating dataset
    train_sampler = BalancedBatchSampler(args.manifest, split="train", batch_size=2, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=0, pin_memory=True,
        collate_fn=collate_simple,
    )

    # --- Model ---
    model = FlowNSFW(
        dim=args.dim, num_heads=args.num_heads,
        num_temporal_layers=args.num_temporal_layers,
        topk_global=args.topk_global,
        flow_backend=args.flow_backend,
        temporal_backend=args.temporal_backend,
        d_state=args.d_state,
        ssm_expand=args.ssm_expand,
        sparse_detect=args.sparse_detect,
    ).to(device)
    counts = model.count_parameters()
    print(f"[flow-nsfw] params={counts['total']/1e6:.2f}M")

    # Report SSM backend
    from flow_nsfw.ssm_backend import SSM_BACKEND, HAS_MAMBA_SSM
    print(f"[flow-nsfw] ssm_backend={SSM_BACKEND} has_cuda={HAS_MAMBA_SSM}")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    start_step = 0

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"] + 1
        print(f"[resume] step={start_step}")

    # --- Training ---
    weights = LossWeights()  # Updated defaults: video_cls=3.0, flow_consistency=0.3, flow_smoothness=0.05
    detection_weight = 2.0  # Weight for simple detection loss
    total_steps = args.epochs * len(train_loader)
    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["step", "lr", "L_total", "L_detection",
                                       "L_video_cls", "L_temporal",
                                       "L_flow_consistency", "L_flow_smoothness",
                                       "elapsed_s"]).writeheader()

    for step in range(start_step, total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        frames = batch["frames"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        video_labels = batch["video_label"].to(device)
        gt_boxes = batch["boxes"]  # List of B lists of T tensors
        B, T = frames.shape[:2]

        lr = _cosine_lr(step, total_steps, args.warmup, args.lr)
        _set_lr(optim, lr)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(frames)

            # Video classification loss
            vcl, vcl_val = video_cls_loss(out["video_cls"], video_labels, weights.video_cls)

            # Temporal smoothness on detection boxes
            tcl, tcl_val = temporal_box_loss(out["decoded"], B, T, weights.temporal)

            # Flow consistency and smoothness losses
            from flow_nsfw.losses import (
                flow_consistency_loss, flow_smoothness_loss, simple_detection_loss
            )
            flow_fwd = out.get("flow_fwd")
            flow_bwd = out.get("flow_bwd")
            fcl, fcl_val = flow_consistency_loss(flow_fwd, flow_bwd, weights.flow_consistency)
            fsl, fsl_val = flow_smoothness_loss(flow_fwd, weights.flow_smoothness)

            # Detection loss with YOLO pseudo-labels
            det_l, det_l_val = simple_detection_loss(out["decoded"], gt_boxes, B, T, detection_weight)

            total = vcl + tcl + fcl + fsl + det_l
            logs = {
                "L_video_cls": vcl_val,
                "L_temporal": tcl_val,
                "L_flow_consistency": fcl_val,
                "L_flow_smoothness": fsl_val,
                "L_detection": det_l_val,
            }

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            msg = f"[flow-nsfw] step={step:5d} lr={lr:.2e} total={total.item():.4f}"
            for k, v in logs.items():
                msg += f" {k}={v:.4f}"
            print(msg, flush=True)
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                row = {"step": step, "lr": f"{lr:.6e}", "L_total": f"{total.item():.6f}",
                       "elapsed_s": f"{elapsed:.1f}"}
                row.update({k: f"{v:.6f}" for k, v in logs.items()})
                csv.DictWriter(f, fieldnames=["step", "lr", "L_total", "L_detection",
                                               "L_video_cls", "L_temporal",
                                               "L_flow_consistency", "L_flow_smoothness",
                                               "elapsed_s"]).writerow(row)

        if (step + 1) % args.ckpt_every == 0:
            torch.save({"step": step, "model": model.state_dict(), "optim": optim.state_dict()},
                       out_dir / f"ckpt_{step+1:05d}.pt")
            print(f"[ckpt] saved step={step}")

    # Final save
    torch.save({"step": total_steps, "model": model.state_dict(), "optim": optim.state_dict()},
               out_dir / "final.pt")
    print(f"[done] final model saved to {out_dir / 'final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
