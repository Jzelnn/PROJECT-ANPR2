import os
import numpy as np
import torch

# Pastikan setting CPU ARMv8 Cortex-A72 aktif
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TORCH_DISABLE_XNNPACK"] = "1"

if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'nnpack'):
    torch.backends.nnpack.enabled = False

from ultralytics import YOLO

test_list = [
    ("1. Vehicle Model", "models/vehicle/best.pt", 384),
    ("2. Plate Best Model", "models/plate/best.pt", 256),
    ("3. Plate Legacy Model", "models/plate/legacy.pt", 256),
    ("4. Body Type Model", "models/body_type/best.pt", 192),
    ("5. Character Model", "models/character/best.pt", 160)
]

print("\n" + "=" * 55)
print("  DIAGNOSA MODEL YOLO DI RASPBERRY PI CORTEX-A72")
print("=" * 55, flush=True)

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
