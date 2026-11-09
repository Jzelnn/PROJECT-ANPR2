import os
import sys

os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["USE_XNNPACK"] = "0"
os.environ["TORCH_DISABLE_XNNPACK"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)
if hasattr(torch.backends, 'mkldnn'): torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'nnpack'): torch.backends.nnpack.enabled = False
if hasattr(torch.backends, 'xnnpack'): torch.backends.xnnpack.enabled = False

print("\n" + "=" * 55)
print("  BEDAH DETAIL TAHAPAN YOLO DI CORTEX-A72")
print("=" * 55, flush=True)

# 1. Tes Torchvision NMS
print("[1] Uji torchvision.ops.nms...", flush=True)
try:
    import torchvision
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [12.0, 12.0, 22.0, 22.0]])
    scores = torch.tensor([0.9, 0.8])
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print(f"    ✅ torchvision.ops.nms BERHASIL! Keep: {keep.tolist()}", flush=True)
except Exception as e:
    print(f"    ❌ torchvision.ops.nms GAGAL: {e}", flush=True)

# 2. Tes OpenCV Resize
print("[2] Uji OpenCV resize...", flush=True)
try:
    import cv2
    img = np.zeros((384, 384, 3), dtype=np.uint8)
    res = cv2.resize(img, (256, 256))
    print("    ✅ OpenCV resize BERHASIL!", flush=True)
except Exception as e:
    print(f"    ❌ OpenCV resize GAGAL: {e}", flush=True)

# 3. Load YOLO Model
from ultralytics import YOLO
path = "models/vehicle/best.pt"
print(f"[3] Load YOLO model {path}...", flush=True)
m = YOLO(path)
print("    ✅ Model loaded!", flush=True)

# 4. Tes Pure PyTorch Model Forward (Tanpa Ultralytics predict wrapper)
print("[4] Uji Pure PyTorch Forward Pass (m.model.eval(), fp32 tensor)...", flush=True)
try:
    dummy_tensor = torch.zeros((1, 3, 384, 384), dtype=torch.float32)
    m.model.eval()
    with torch.no_grad():
        out = m.model(dummy_tensor)
    print(f"    ✅ Pure PyTorch Forward Pass BERHASIL! out type: {type(out)}", flush=True)
except Exception as e:
    print(f"    ❌ Forward Pass GAGAL: {e}", flush=True)

# 5. Tes Ultralytics Post-processing / NMS
print("[5] Uji Ultralytics non_max_suppression...", flush=True)
try:
    from ultralytics.utils.ops import non_max_suppression
    preds = non_max_suppression(out[0] if isinstance(out, (list, tuple)) else out)
    print(f"    ✅ non_max_suppression BERHASIL!", flush=True)
except Exception as e:
    print(f"    ❌ non_max_suppression GAGAL: {e}", flush=True)

# 6. Tes m.predict dengan numpy array
print("[6] Uji m.predict(img, half=False, device='cpu')...", flush=True)
try:
    results = m.predict(img, half=False, device='cpu', verbose=False)
    print("    ✅ m.predict BERHASIL!", flush=True)
except Exception as e:
    print(f"    ❌ m.predict GAGAL: {e}", flush=True)

print("\n" + "=" * 55)
print("  SELESAI BEDAH TAHAPAN")
print("=" * 55 + "\n", flush=True)
