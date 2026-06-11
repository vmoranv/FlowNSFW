#!/bin/bash
# Fix corrupted image handling in FlowNSFW A10 package

cd "$(dirname "$0")"

echo "Patching pseudo_labeler.py for corrupted image handling..."
cat > src/flow_nsfw/pseudo_labeler.py.patch << 'EOF'
--- a/src/flow_nsfw/pseudo_labeler.py
+++ b/src/flow_nsfw/pseudo_labeler.py
@@ -55,9 +55,14 @@ def _decode_avif_to_numpy(path: Path) -> np.ndarray:
 def _decode_image(path: Path) -> np.ndarray:
-    """Read any image as RGB numpy array."""
+    """Read any image as RGB numpy array. Returns None if corrupted."""
     import cv2
-    data = np.fromfile(str(path), dtype=np.uint8)
-    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
-    if im is not None:
-        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
-    # Try ffmpeg for AVIF / exotic formats
-    return _decode_avif_to_numpy(path)
+    try:
+        data = np.fromfile(str(path), dtype=np.uint8)
+        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
+        if im is not None:
+            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
+        # Try ffmpeg for AVIF / exotic formats
+        return _decode_avif_to_numpy(path)
+    except Exception as e:
+        # Corrupted file, return None to skip
+        print(f"[WARN] Skipping corrupted image {path}: {e}")
+        return None
EOF

echo "Patching data.py for None image handling..."
cat > src/flow_nsfw/data.py.patch << 'EOF'
--- a/src/flow_nsfw/data.py
+++ b/src/flow_nsfw/data.py
@@ -113,8 +113,12 @@ class VideoClipDataset(Dataset):
         for i in range(T):
             path = frame_paths[start + i * stride]
             img = _read_img(Path(path))
-            orig_h, orig_w = img.shape[:2]
-            if img.shape[:2] != (h, w):
+            if img is None:
+                # Corrupted frame, use black placeholder
+                img = np.zeros((h, w, 3), dtype=np.uint8)
+                orig_h, orig_w = h, w
+            else:
+                orig_h, orig_w = img.shape[:2]
+                if img.shape[:2] != (h, w):
                 img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
             frames[i] = img
EOF

# Apply patches
sed -i 's/def _decode_image(path: Path) -> np.ndarray:/def _decode_image(path: Path) -> np.ndarray | None:/' src/flow_nsfw/pseudo_labeler.py
sed -i '57,63d' src/flow_nsfw/pseudo_labeler.py
sed -i '56a\    """Read any image as RGB numpy array. Returns None if corrupted."""\n    import cv2\n    try:\n        data = np.fromfile(str(path), dtype=np.uint8)\n        im = cv2.imdecode(data, cv2.IMREAD_COLOR)\n        if im is not None:\n            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)\n        # Try ffmpeg for AVIF / exotic formats\n        return _decode_avif_to_numpy(path)\n    except Exception as e:\n        # Corrupted file, return None to skip\n        print(f"[WARN] Skipping corrupted image {path}: {e}")\n        return None' src/flow_nsfw/pseudo_labeler.py

# Fix data.py
sed -i '115,118d' src/flow_nsfw/data.py
sed -i '114a\            img = _read_img(Path(path))\n            if img is None:\n                # Corrupted frame, use black placeholder\n                img = np.zeros((h, w, 3), dtype=np.uint8)\n                orig_h, orig_w = h, w\n            else:\n                orig_h, orig_w = img.shape[:2]\n                if img.shape[:2] != (h, w):\n                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)' src/flow_nsfw/data.py

echo "✅ Patches applied. Now run: bash train.sh"
