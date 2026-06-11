#!/usr/bin/env python3
"""Fix corrupted image handling in FlowNSFW A10 package."""
import sys
from pathlib import Path

def fix_pseudo_labeler():
    """Patch pseudo_labeler.py to return None for corrupted images."""
    path = Path("src/flow_nsfw/pseudo_labeler.py")
    content = path.read_text()

    old = '''def _decode_image(path: Path) -> np.ndarray:
    """Read any image as RGB numpy array."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is not None:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Try ffmpeg for AVIF / exotic formats
    return _decode_avif_to_numpy(path)'''

    new = '''def _decode_image(path: Path) -> np.ndarray:
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
        return None'''

    if old not in content:
        print("⚠️  pseudo_labeler.py: pattern not found, skipping")
        return False

    content = content.replace(old, new)
    path.write_text(content)
    print("✅ Fixed pseudo_labeler.py")
    return True

def fix_data():
    """Patch data.py to handle None from _read_img."""
    path = Path("src/flow_nsfw/data.py")
    content = path.read_text()

    old = '''            path = frame_paths[start + i * stride]
            img = _read_img(Path(path))
            orig_h, orig_w = img.shape[:2]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            frames[i] = img'''

    new = '''            path = frame_paths[start + i * stride]
            img = _read_img(Path(path))
            if img is None:
                # Corrupted frame, use black placeholder
                img = np.zeros((h, w, 3), dtype=np.uint8)
                orig_h, orig_w = h, w
            else:
                orig_h, orig_w = img.shape[:2]
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            frames[i] = img'''

    if old not in content:
        print("⚠️  data.py: pattern not found, skipping")
        return False

    content = content.replace(old, new)
    path.write_text(content)
    print("✅ Fixed data.py")
    return True

if __name__ == "__main__":
    print("=== FlowNSFW A10 Corruption Fix ===")
    ok1 = fix_pseudo_labeler()
    ok2 = fix_data()
    if ok1 and ok2:
        print("\n✅ All patches applied successfully!")
        print("Now run: bash train.sh")
        sys.exit(0)
    else:
        print("\n⚠️  Some patches failed. Check manually.")
        sys.exit(1)
