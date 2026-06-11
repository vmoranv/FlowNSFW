#!/bin/bash
# FlowNSFW V8 — A10 cloud training launcher
# Usage: bash train_v8_a10.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# A10 has 24GB VRAM — can use larger resolution
python3 -u scripts/train.py \
  --manifest datasets/manifest_v4_clean_wsl.json \
  --temporal-backend mamba --d-state 16 --ssm-expand 2 \
  --sparse-detect \
  --epochs 80 --batch-size 1 --clip-len 8 --lr 2e-4 \
  --dim 128 --num-heads 4 --num-temporal-layers 3 --topk-global 64 \
  --log-every 10 --ckpt-every 1000 \
  --out runs/v8_a10 --bf16 --device cuda

# Eval when done
echo "=== Training complete, running eval ==="
python3 -u scripts/eval_multi_res.py \
  --ckpt runs/v8_a10/final.pt \
  --manifest datasets/manifest_v4_clean_wsl.json \
  --temporal-backend mamba --d-state 16 --ssm-expand 2 --sparse-detect \
  --resolutions 160 240 320 480 640 \
  --device cuda
