#!/usr/bin/env python3
"""FlowNSFW 生产模型评测脚本 — 支持所有优化配置.

Usage:
    python scripts/eval_production.py --ckpt runs/prod_mamba3_full/final.pt --manifest datasets/manifest_v4_clean_wsl.json --mode mamba3_full --output eval_results.json
"""

import argparse, json, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch, numpy as np, cv2
from flow_nsfw import FlowNSFW
from flow_nsfw.data import _read_img
from flow_nsfw.memory_opt import optimize_memory_layout

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_video_frames(frame_dir: str, clip_len: int = 4, stride: int = 2) -> list:
    """加载一个视频的所有帧路径."""
    p = Path(frame_dir)
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    if len(files) < clip_len * stride:
        return None
    return files


def infer_clip(model, frame_paths, start, clip_len, stride, resolution, device):
    """对一个clip做推理."""
    frames = []
    for i in range(clip_len):
        idx = start + i * stride
        if idx >= len(frame_paths):
            return None
        img = _read_img(frame_paths[idx])
        if img is None:
            img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        elif img.shape[:2] != (resolution, resolution):
            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        frames.append(img)

    frames_np = np.stack(frames).astype(np.float32) / 255.0
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0)  # (1,T,3,H,W)
    frames_t = frames_t.to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(frames_t)

    probs = torch.softmax(out["video_cls"].float(), -1)
    return probs[0, 1].item()  # NSFW confidence


def evaluate_video(model, frame_dir, clip_len, stride, resolution, device):
    """对单个视频做滑动窗口评测."""
    files = load_video_frames(frame_dir, clip_len, stride)
    if files is None:
        return None

    T = len(files)
    if T < clip_len * stride:
        return None

    # 多个clip采样
    confs = []
    max_start = max(0, T - clip_len * stride)

    for start in range(0, max_start + 1, stride):
        conf = infer_clip(model, files, start, clip_len, stride, resolution, device)
        if conf is not None:
            confs.append(conf)

    if not confs:
        return None

    return {
        "max_conf": max(confs),
        "mean_conf": sum(confs) / len(confs),
        "n_clips": len(confs),
        "verdict": "NSFW" if max(confs) > 0.5 else "SFW",
    }


def main():
    ap = argparse.ArgumentParser(description="FlowNSFW 生产模型评测")
    ap.add_argument("--ckpt", required=True, help="模型权重路径")
    ap.add_argument("--manifest", required=True, help="评测数据manifest")
    ap.add_argument("--output", default="eval_results.json", help="结果输出文件")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="mamba3_full",
                    choices=["mamba3_full", "no_encoder_full", "mamba2_baseline", "mamba2_full", "mamba2_sparse"])
    ap.add_argument("--clip-len", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--temporal-backend", default="mamba")
    ap.add_argument("--limit", type=int, default=0, help="限制评测视频数")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"Loading model: {args.ckpt}")
    model_kwargs = {
        "dim": args.dim,
        "num_heads": 4,
        "num_temporal_layers": 3,
        "topk_global": 64,
        "temporal_backend": args.temporal_backend,
        "sparse_detect": True,
    }

    if "mamba3" in args.mode:
        model_kwargs["ssm_backend"] = "mamba3"
    else:
        model_kwargs["ssm_backend"] = "mamba2"

    if args.mode.startswith("no_encoder"):
        model_kwargs["no_encoder"] = True
        model_kwargs["patch_size"] = 16

    if "full" in args.mode or "sparse" in args.mode:
        model_kwargs["motion_gate"] = True

    if "sparse" in args.mode:
        model_kwargs["motion_sparse_token"] = True
        model_kwargs["sparse_topk"] = 200

    model = FlowNSFW(**model_kwargs).to(device)
    model = optimize_memory_layout(model)
    model.eval()

    # 加载权重
    ck = torch.load(args.ckpt, map_location=device)
    sd = ck.get("model", ck)
    model.load_state_dict(sd, strict=False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.2f}M")
    print(f"  mode: {args.mode}")
    print(f"  resolution: {args.resolution}")

    # 加载评测数据
    with open(args.manifest) as f:
        manifest = json.load(f)

    videos = [v for v in manifest if v.get("split") == "val"]
    if args.limit > 0:
        videos = videos[:args.limit]

    print(f"\nEvaluating {len(videos)} videos...")

    results = []
    correct, total = 0, 0
    nsfw_ok, nsfw_total = 0, 0
    sfw_ok, sfw_total = 0, 0
    t0 = time.time()

    for i, v in enumerate(videos):
        frames = [f for f in v["frames"] if Path(f).exists()]
        if len(frames) < args.clip_len * args.stride:
            print(f"[{i+1}/{len(videos)}] {v['id']}: SKIP (frames={len(frames)})")
            continue

        frame_dir = str(Path(frames[0]).parent)
        result = evaluate_video(
            model, frame_dir, args.clip_len, args.stride,
            args.resolution, device,
        )

        if result is None:
            print(f"[{i+1}/{len(videos)}] {v['id']}: SKIP (no clips)")
            continue

        gt = "NSFW" if v.get("label") == 1 else "SFW"
        ok = "OK" if result["verdict"] == gt else "FAIL"
        result["video_id"] = v.get("id", f"vid_{i}")
        result["gt"] = gt
        result["ok"] = result["verdict"] == gt
        result["elapsed"] = time.time() - t0

        total += 1
        if result["ok"]:
            correct += 1
        if v.get("label") == 1:
            nsfw_total += 1
            if result["verdict"] == "NSFW":
                nsfw_ok += 1
        else:
            sfw_total += 1
            if result["verdict"] == "SFW":
                sfw_ok += 1

        results.append(result)

        print(f"[{i+1}/{len(videos)}] {result['video_id'][:30]}: "
              f"{result['verdict']:5s} (conf={result['max_conf']:.3f}) "
              f"GT={gt} {ok}")

    # 汇总
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Videos evaluated: {total}")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(1,total):.2f}s/video)")

    if total > 0:
        acc = correct / total * 100
        nsfw_recall = nsfw_ok / max(1, nsfw_total) * 100
        sfw_acc = sfw_ok / max(1, sfw_total) * 100

        print(f"Accuracy: {acc:.1f}% ({correct}/{total})")
        print(f"NSFW Recall: {nsfw_recall:.1f}% ({nsfw_ok}/{nsfw_total})")
        print(f"SFW Accuracy: {sfw_acc:.1f}% ({sfw_ok}/{sfw_total})")

    # 保存结果
    summary = {
        "mode": args.mode,
        "resolution": args.resolution,
        "params_m": n_params,
        "total": total,
        "accuracy": acc if total > 0 else 0,
        "nsfw_recall": nsfw_recall if total > 0 else 0,
        "sfw_accuracy": sfw_acc if total > 0 else 0,
        "elapsed_s": elapsed,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
