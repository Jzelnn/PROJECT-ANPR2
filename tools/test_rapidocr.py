"""
Tes RapidOCR (ONNX Runtime) di Raspberry Pi 4 (Cortex-A72).

Pasang dulu (di dalam venv, butuh internet sekali saja):
    pip install onnxruntime rapidocr-onnxruntime

Jalankan:
    python tools/test_rapidocr.py [path_gambar_plat.jpg]
Tanpa argumen, dipakai gambar plat sintetis "B 902 JRN".
"""
import os
import sys
import time

os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import cv2

print("[1] import onnxruntime...", flush=True)
import onnxruntime as ort
print(f"    OK versi {ort.__version__}", flush=True)

print("[2] import rapidocr_onnxruntime...", flush=True)
from rapidocr_onnxruntime import RapidOCR
print("    OK", flush=True)

print("[3] inisialisasi RapidOCR...", flush=True)
t0 = time.time()
engine = RapidOCR()
print(f"    OK init={(time.time() - t0) * 1000:.0f}ms", flush=True)

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    img = cv2.imread(sys.argv[1])
    print(f"[4] gambar: {sys.argv[1]} shape={img.shape}", flush=True)
else:
    img = np.full((90, 270, 3), 245, dtype=np.uint8)
    cv2.putText(img, "B 902 JRN", (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (10, 10, 10), 4)
    print("[4] gambar sintetis 'B 902 JRN'", flush=True)

for i in range(3):
    t = time.time()
    result, _ = engine(img)
    ms = (time.time() - t) * 1000
    texts = [(r[1], round(float(r[2]), 3)) for r in (result or [])]
    print(f"[5.{i + 1}] {ms:.0f}ms -> {texts}", flush=True)

print("\nSELESAI: RapidOCR jalan di CPU ini tanpa Illegal instruction.", flush=True)
