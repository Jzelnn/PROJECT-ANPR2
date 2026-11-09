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

# Matikan CPU AMP / Autocast agar tidak menyentuh instruksi ARMv8.2 BFLOAT16/FP16
class NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass

if hasattr(torch, 'autocast'):
    torch.autocast = lambda *args, **kwargs: NullContext()
if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    torch.cuda.amp.autocast = lambda *args, **kwargs: NullContext()
if hasattr(torch.cpu, 'amp') and hasattr(torch.cpu.amp, 'autocast'):
    torch.cpu.amp.autocast = lambda *args, **kwargs: NullContext()

from ultralytics import YOLO

path = "models/vehicle/best.pt"
print(f"Loading {path}...", flush=True)
m = YOLO(path)
raw_img = np.zeros((384, 384, 3), dtype=np.uint8)

print("\n[TEST FINAL] Menjalankan m.predict dengan amp=False & imgsz=384...", flush=True)
try:
    res = m.predict(raw_img, imgsz=384, verbose=False, device='cpu', amp=False)
    print(f"    🎉🎉🎉 SUKSES BESAR! m.predict BERHASIL DI CORTEX-A72! 🎉🎉🎉", flush=True)
    print(f"    Jumlah hasil deteksi: {len(res)}", flush=True)
    print(f"    Boxes: {len(res[0].boxes)}", flush=True)
except Exception as e:
    import traceback
    print(f"    ❌ Error: {e}", flush=True)
    traceback.print_exc()
