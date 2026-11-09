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

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor

path = "models/vehicle/best.pt"
print(f"Loading {path}...", flush=True)
m = YOLO(path)
raw_img = np.zeros((384, 384, 3), dtype=np.uint8)

print("\n[1] Memeriksa struktur output m.model(im)...", flush=True)
im = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
with torch.no_grad():
    out = m.model(im)
if isinstance(out, (list, tuple)):
    print(f"    Output adalah {type(out)} dengan panjang {len(out)}")
    for i, item in enumerate(out):
        if hasattr(item, 'shape'):
            print(f"    Item [{i}] shape: {item.shape}, dtype: {item.dtype}")
else:
    print(f"    Output shape: {out.shape}")

print("\n[2] Inisialisasi DetectionPredictor...", flush=True)
try:
    overrides = dict(conf=0.25, iou=0.45, device='cpu', verbose=False)
    predictor = DetectionPredictor(overrides=overrides)
    predictor.setup_model(model=m.model, verbose=False)
    print("    ✅ DetectionPredictor setup OK!", flush=True)
    
    print("\n[3] Jalankan predictor.preprocess...", flush=True)
    prep = predictor.preprocess([raw_img])
    print(f"    ✅ Preprocess OK! Shape: {prep.shape}", flush=True)
    
    print("\n[4] Jalankan predictor.inference...", flush=True)
    inf = predictor.inference(prep)
    print(f"    ✅ Inference OK!", flush=True)
    
    print("\n[5] Jalankan predictor.postprocess...", flush=True)
    post = predictor.postprocess(inf, prep, [raw_img])
    print(f"    ✅ Postprocess OK! Jumlah deteksi: {len(post)}", flush=True)
    
    print("\n[6] Jalankan predictor([raw_img])...", flush=True)
    res = predictor([raw_img])
    print(f"    ✅ Predictor pipeline lengkap SUKSES! Boxes: {len(res[0].boxes)}", flush=True)

except Exception as e:
    import traceback
    print(f"    ❌ Error: {e}", flush=True)
    traceback.print_exc()

print("\n" + "=" * 55)
print("  SELESAI PENGUJIAN PREDICTOR")
print("=" * 55 + "\n", flush=True)
