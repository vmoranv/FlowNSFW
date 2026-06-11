#!/bin/bash
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
echo "=== FlowNSFW V10 Mamba Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
python3 -u scripts/train.py \
  --manifest manifest.json \
  --temporal-backend mamba --d-state 16 --ssm-expand 2 --sparse-detect \
  --epochs 100 --batch-size 1 --clip-len 8 --lr 2e-4 \
  --dim 128 --num-heads 4 --num-temporal-layers 3 --topk-global 64 \
  --log-every 10 --ckpt-every 2000 \
  --out runs/v10 --bf16 --device cuda
echo "=== Training complete, running eval ==="
python3 -u scripts/eval_multi_res.py \
  --ckpt runs/v10/final.pt --manifest manifest.json \
  --temporal-backend mamba --d-state 16 --ssm-expand 2 --sparse-detect \
  --resolutions 160 240 320 480 640 --device cuda
