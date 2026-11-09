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

path = "models/vehicle/best.pt"
print(f"Loading {path}...", flush=True)
m = YOLO(path)
print("Model loaded. Initializing Predictor...", flush=True)

# Uji Preprocessing manual Ultralytics
print("\n[A] Uji Preprocess Gambar Numpy -> Tensor...", flush=True)
raw_img = np.zeros((384, 384, 3), dtype=np.uint8)
im = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print(f"    Tensor shape: {im.shape}, dtype: {im.dtype}", flush=True)

# Uji Forward
print("\n[B] Uji Forward Pass m.model(im)...", flush=True)
with torch.no_grad():
    preds = m.model(im)
print(f"    Forward OK! Preds type: {type(preds)}", flush=True)

# Uji Predictor Postprocess
print("\n[C] Uji Predictor Internal Ultralytics...", flush=True)
try:
    # Buat predictor
    overrides = dict(conf=0.25, iou=0.45, device='cpu', verbose=False)
    predictor = m._smart_load('predict')(overrides=overrides)
    predictor.setup_model(model=m.model, verbose=False)
    print("    Predictor setup OK!", flush=True)
    
    print("\n[D] Uji predictor.preprocess...", flush=True)
    prep = predictor.preprocess([raw_img])
    print(f"    Preprocess OK! Shape: {prep.shape}", flush=True)
    
    print("\n[E] Uji predictor.inference...", flush=True)
    inf = predictor.inference(prep)
    print(f"    Inference OK!", flush=True)
    
    print("\n[F] Uji predictor.postprocess...", flush=True)
    post = predictor.postprocess(inf, prep, [raw_img])
    print(f"    Postprocess OK! Result count: {len(post)}", flush=True)

except Exception as e:
    import traceback
    print(f"    ❌ Error: {e}", flush=True)
    traceback.print_exc()

print("\n" + "=" * 55)
print("  SELESAI BEDAH INTERNAL PREDICTOR")
print("=" * 55 + "\n", flush=True)
