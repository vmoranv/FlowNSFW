#!/bin/bash
cd /mnt/d/cumhub/flow-nsfw
mkdir -p runs/flow_nsfw_v7_balanced
exec python3 -u scripts/train.py \
  --manifest datasets/manifest_v4_clean_wsl.json \
  --temporal-backend mamba --d-state 16 --ssm-expand 2 \
  --sparse-detect --epochs 80 --clip-len 4 --lr 2e-4 \
  --dim 128 --num-heads 4 --num-temporal-layers 3 --topk-global 64 \
  --log-every 10 --ckpt-every 1000 \
  --out runs/flow_nsfw_v7_balanced --bf16 --device cuda
