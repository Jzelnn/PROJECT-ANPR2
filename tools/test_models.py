import os
import sys

# WAJIB: Atur variabel lingkungan SEBELUM numpy / torch di-import!
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

# Matikan CPU AMP / Autocast agar tidak menyentuh instruksi ARMv8.2 BFLOAT16/FP16 pada Cortex-A72
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
from ultralytics.engine.model import Model

orig_model_predict = Model.predict
def safe_arm_predict(self, *args, **kwargs):
    kwargs['device'] = 'cpu'
    kwargs['amp'] = False
    return orig_model_predict(self, *args, **kwargs)
Model.predict = safe_arm_predict

print("\n" + "=" * 55)
print("  DIAGNOSA MODEL YOLO DI RASPBERRY PI CORTEX-A72")
print("=" * 55, flush=True)
print(f"PyTorch Version: {torch.__version__}")
print(f"Threads: {torch.get_num_threads()}")

test_list = [
    ("1. Vehicle Model", "models/vehicle/best.pt", 384),
    ("2. Plate Best Model", "models/plate/best.pt", 256),
    ("3. Plate Legacy Model", "models/plate/legacy.pt", 256),
    ("4. Body Type Model", "models/body_type/best.pt", 192),
    ("5. Character Model", "models/character/best.pt", 160)
]

for name, path, sz in test_list:
    if os.path.exists(path):
        print(f"\n[TEST] {name} ({path})...", flush=True)
        try:
            m = YOLO(path)
            res = m.predict(np.zeros((sz, sz, 3), dtype=np.uint8), imgsz=sz, verbose=False)
            print(f"       ✅ {name}: 100% SUKSES DAN KOMPATIBEL! (Detections: {len(res[0].boxes)})", flush=True)
        except Exception as e:
            print(f"       ❌ {name} GAGAL: {e}", flush=True)
    else:
        print(f"[SKIP] File {path} tidak ditemukan.", flush=True)

print("\n" + "=" * 55)
print("  SEMUA PENGUJIAN SELESAI!")
print("=" * 55 + "\n", flush=True)
