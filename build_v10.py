"""Build V10 complete package — Python-based, no shell issues on NTFS."""
import os, shutil, json
from pathlib import Path

ROOT = Path("D:/cumhub/flow-nsfw")
PKG = ROOT / "flow_nsfw_v10"
SRC = PKG / "src" / "flow_nsfw"
SCRIPTS = PKG / "scripts"
RUNS = PKG / "runs"

# Nuke and recreate
if PKG.exists():
    shutil.rmtree(PKG, ignore_errors=True)
PKG.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)
SCRIPTS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

# Copy fixed source
for f in (ROOT / "src" / "flow_nsfw").glob("*.py"):
    shutil.copy2(str(f), str(SRC / f.name))
print(f"src: {len(list(SRC.glob('*.py')))} files")

# Copy scripts
for f in ["train.py", "eval_multi_res.py"]:
    shutil.copy2(str(ROOT / "scripts" / f), str(SCRIPTS / f))
print(f"scripts: {len(list(SCRIPTS.glob('*.py')))} files")

# Copy clean data (copytree handles dirs)
data_src = ROOT / "flow_nsfw_a10_package" / "data"
data_dst = PKG / "data"
if data_src.exists():
    shutil.copytree(str(data_src), str(data_dst), symlinks=False)
    jpg_count = sum(1 for _ in data_dst.rglob("*.jpg"))
    print(f"data: {jpg_count} frames")

# Copy manifest
manifest_src = ROOT / "flow_nsfw_a10_package" / "manifest.json"
shutil.copy2(str(manifest_src), str(PKG / "manifest.json"))
print("manifest: copied")

# Write train.sh
train_sh = """#!/bin/bash
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
echo "=== FlowNSFW V10 Mamba Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
python3 -u scripts/train.py \\
  --manifest manifest.json \\
  --temporal-backend mamba --d-state 16 --ssm-expand 2 --sparse-detect \\
  --epochs 100 --batch-size 1 --clip-len 8 --lr 2e-4 \\
  --dim 128 --num-heads 4 --num-temporal-layers 3 --topk-global 64 \\
  --log-every 10 --ckpt-every 2000 \\
  --out runs/v10 --bf16 --device cuda
echo "=== Training complete, running eval ==="
python3 -u scripts/eval_multi_res.py \\
  --ckpt runs/v10/final.pt --manifest manifest.json \\
  --temporal-backend mamba --d-state 16 --ssm-expand 2 --sparse-detect \\
  --resolutions 160 240 320 480 640 --device cuda
"""
(PKG / "train.sh").write_text(train_sh)
print("train.sh: written")

# Verify
print(f"\n=== V10 Package ===")
for root, dirs, files in os.walk(str(PKG)):
    level = root.replace(str(PKG), "").count(os.sep)
    if level <= 3:
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/ ({len(dirs)} dirs, {len(files)} files)")
        if level >= 2:
            dirs.clear()

total_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(str(PKG)) for f in fs) / 1024 / 1024
print(f"\nTotal size: {total_mb:.0f} MB")
