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
if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'nnpack'):
    torch.backends.nnpack.enabled = False
if hasattr(torch.backends, 'xnnpack'):
    torch.backends.xnnpack.enabled = False

print("\n" + "=" * 55)
print("  DIAGNOSA MODEL YOLO DI RASPBERRY PI CORTEX-A72")
print("=" * 55, flush=True)
print(f"PyTorch Version: {torch.__version__}")
print(f"Threads: {torch.get_num_threads()}")

# Tes konvolusi dasar
print("\n[TEST AWAL] Uji PyTorch Conv2d pada Cortex-A72...", flush=True)
try:
    conv = torch.nn.Conv2d(3, 16, 3)
    dummy_t = torch.randn(1, 3, 32, 32)
    out_t = conv(dummy_t)
    print(f"       ✅ Conv2d Berhasil! Output shape: {list(out_t.shape)}")
except Exception as e:
    print(f"       ❌ Conv2d Gagal: {e}")

from ultralytics import YOLO

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
            print(f"       Load OK! Menjalankan predict {sz}x{sz}...", flush=True)
            res = m.predict(np.zeros((sz, sz, 3), dtype=np.uint8), verbose=False)
            print(f"       ✅ {name}: 100% SUKSES DAN KOMPATIBEL!", flush=True)
        except Exception as e:
            print(f"       ❌ {name} GAGAL: {e}", flush=True)
    else:
        print(f"[SKIP] File {path} tidak ditemukan.", flush=True)

print("\n" + "=" * 55)
print("  PENGUJIAN SELESAI!")
print("=" * 55 + "\n", flush=True)
