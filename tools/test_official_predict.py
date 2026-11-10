import os
import sys

# Konfigurasi CPU ARMv8 Cortex-A72
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["USE_XNNPACK"] = "0"
os.environ["TORCH_DISABLE_XNNPACK"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["DISABLE_PADDLE"] = "1"

import numpy as np
import cv2
import torch

torch.set_num_threads(1)
if hasattr(torch.backends, 'mkldnn'): torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'nnpack'): torch.backends.nnpack.enabled = False
if hasattr(torch.backends, 'xnnpack'): torch.backends.xnnpack.enabled = False

# Matikan CPU AMP / Autocast
class NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass

if hasattr(torch, 'autocast'): torch.autocast = lambda *args, **kwargs: NullContext()
if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'): torch.cuda.amp.autocast = lambda *args, **kwargs: NullContext()
if hasattr(torch.cpu, 'amp') and hasattr(torch.cpu.amp, 'autocast'): torch.cpu.amp.autocast = lambda *args, **kwargs: NullContext()

from ultralytics import YOLO

print("\n" + "=" * 55)
print("  UJI ULTRALYTICS ASLI (OFFICIAL PREDICT) DI CORTEX-A72")
print("=" * 55, flush=True)

test_list = [
    ("Vehicle Model", "models/vehicle/best.pt", (384, 384)),
    ("Plate Model", "models/plate/best.pt", (256, 256)),
    ("Body Type Model", "models/body_type/best.pt", (192, 192)),
    ("Character Model", "models/character/best.pt", (80, 240)) # Aspek rasio plat asli!
]

for name, path, shape in test_list:
    if os.path.exists(path):
        print(f"\n[TEST] {name} ({path}) shape={shape}...", flush=True)
        try:
            m = YOLO(path)
            dummy = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
            # Jalankan predict ASLI bawaan Ultralytics
            res = m.predict(dummy, device='cpu', verbose=False)
            if hasattr(res[0], 'boxes') and res[0].boxes is not None:
                print(f"       ✅ {name}: SUKSES BESAR! Boxes count: {len(res[0].boxes)}", flush=True)
            elif hasattr(res[0], 'probs') and res[0].probs is not None:
                print(f"       ✅ {name}: SUKSES BESAR! Probs top1: {res[0].probs.top1}", flush=True)
            else:
                print(f"       ✅ {name}: SUKSES BESAR! Output: {type(res[0])}", flush=True)
        except Exception as e:
            print(f"       ❌ {name} GAGAL: {e}", flush=True)
            import traceback
            traceback.print_exc()

print("\n" + "=" * 55)
print("  PENGUJIAN SELESAI!")
print("=" * 55 + "\n", flush=True)
