#!/usr/bin/env python3
"""A10 fix: update pseudo_labeler.py + train.py for V9."""
from pathlib import Path

# Fix pseudo_labeler — PIL fallback
pl_path = Path("src/flow_nsfw/pseudo_labeler.py")
old_pl = """def _decode_image(path: Path) -> np.ndarray:
    """Read any image as RGB numpy array. Returns None if corrupted."""
    import cv2
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if im is not None:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # Try ffmpeg for AVIF / exotic formats
        return _decode_avif_to_numpy(path)
    except Exception as e:
        # Corrupted file, return None to skip
        print(f"[WARN] Skipping corrupted image {path}: {e}")
        return None"""

new_pl = """def _decode_image(path: Path) -> np.ndarray:
    """Read any image as RGB numpy array.

    Decode chain: cv2 → PIL/pillow-avif → ffmpeg. Returns None only if all fail.
    """
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)

    # 1) Try OpenCV (fast path)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is not None:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # 2) Try PIL (handles AVIF if pillow-avif-plugin is installed)
    try:
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(data))
        pil_img = pil_img.convert("RGB")
        return np.array(pil_img)
    except Exception:
        pass

    # 3) Try ffmpeg for exotic / malformed formats
    try:
        return _decode_avif_to_numpy(path)
    except Exception:
        pass

    # All decoders failed — use black placeholder
    return None"""

content = pl_path.read_text()
if old_pl in content:
    content = content.replace(old_pl, new_pl)
    pl_path.write_text(content)
    print("✅ Fixed pseudo_labeler.py (PIL fallback)")
else:
    print("⚠️  pseudo_labeler.py: pattern not found, may already be fixed")

# Fix train.py resolution
tr_path = Path("scripts/train.py")
old_tr = """    if args.clip_len >= 8:
        resolution = (192, 192)  # 8-frame clips need less spatial res to fit VRAM
    else:
        resolution = (256, 256)"""

new_tr = """    if args.clip_len >= 8 and args.batch_size >= 2:
        resolution = (192, 192)
    elif args.clip_len >= 8:
        resolution = (256, 256)  # A10 24GB can handle clip8@256 with batch=1
    else:
        resolution = (320, 320)"""

content2 = tr_path.read_text()
if old_tr in content2:
    content2 = content2.replace(old_tr, new_tr)
    tr_path.write_text(content2)
    print("✅ Fixed train.py (256px for A10)")
else:
    print("⚠️  train.py: pattern not found, may already be fixed")

print("\n✅ V9 fixes applied. Now run: bash train.sh")
