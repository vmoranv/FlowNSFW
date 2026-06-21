#!/bin/bash
# A10 单配置训练 — 推荐生产配置

cd "$(dirname "$0")/.."

MANIFEST="datasets/manifest_train.json"
VAL_MANIFEST="datasets/manifest_val.json"
EPOCHS=30
BATCH=8
RES=320
OUT="runs/prod_mamba2_full"

echo "=========================================="
echo "A10 Production Training"
echo "=========================================="
echo "Config: mamba2_full + all optimizations"
echo "Resolution: ${RES}x${RES}"
echo "Batch size: $BATCH"
echo "Epochs: $EPOCHS"
echo "=========================================="

python3 scripts/train.py \
  --manifest "$MANIFEST" \
  --epochs $EPOCHS \
  --batch-size $BATCH \
  --clip-len 4 \
  --resolution $RES \
  --out "$OUT" \
  --temporal-backend mamba \
  --ssm-backend mamba2 \
  --motion-gate \
  --sparse-detect \
  --log-every 50 \
  --ckpt-every 500

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
echo "Model saved to: $OUT/final.pt"
echo ""
echo "Starting evaluation..."

python3 scripts/eval_production.py \
  --ckpt "$OUT/final.pt" \
  --manifest "$VAL_MANIFEST" \
  --mode mamba2_full \
  --resolution $RES \
  --output "${OUT}/eval_result.json"

echo ""
echo "=========================================="
echo "Results:"
cat "${OUT}/eval_result.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"Accuracy: {d['accuracy']:.1f}%\")
print(f\"NSFW Recall: {d['nsfw_recall']:.1f}%\")
print(f\"SFW Accuracy: {d['sfw_accuracy']:.1f}%\")
print(f\"Inference: {d['elapsed_s']/d['total']:.2f}s/video\")
"
echo "=========================================="
