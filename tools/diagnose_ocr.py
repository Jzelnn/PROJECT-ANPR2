"""
Diagnosa OCR plat di Raspberry Pi.

Pakai:
    SKIP_WARMUP=1 DISABLE_PADDLE=1 python tools/diagnose_ocr.py path/foto_mobil.jpg

Output:
  - waktu tiap tahap
  - bbox plat yang ditemukan + file crop plat (output/diag/plate_crop.jpg)
  - hasil char model pada beberapa imgsz / skala / conf (+ overlay gambar)
  - hasil RapidOCR (recognition-only dan full) pada crop yang sama
Kirim teks output + isi folder output/diag/ ke saya.
"""
import os
import sys
import time

os.environ.setdefault("SKIP_WARMUP", "1")
os.environ.setdefault("DISABLE_PADDLE", "1")
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.chdir(ROOT)

import cv2
import numpy as np

if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
    print("Pakai: python tools/diagnose_ocr.py path/foto_mobil.jpg")
    sys.exit(1)

print("[0] import backend (memuat model)...", flush=True)
t0 = time.time()
import app as A
print(f"    selesai {(time.time() - t0):.1f}s", flush=True)

OUT = os.path.join(ROOT, "output", "diag")
os.makedirs(OUT, exist_ok=True)

img = cv2.imread(sys.argv[1])
print(f"[1] foto {sys.argv[1]} shape={img.shape}", flush=True)

print("[2] run_anpr penuh (foto statis)...", flush=True)
t = time.time()
res = A.run_anpr(img, is_stream=False)
print(f"    total {(time.time() - t) * 1000:.0f}ms", flush=True)
dets = res.get("detections") or []
if not dets:
    print("    TIDAK ADA deteksi kendaraan.")
    sys.exit(0)
d = dets[0]
print(f"    vehicle={d.get('vehicle_type')} body={d.get('body_style')} "
      f"plate_text={d.get('license_plate')!r} ocr_conf={d.get('ocr_confidence')} "
      f"method={d.get('ocr_method')} plate_bbox={d.get('plate_bbox')}", flush=True)

pb = d.get("plate_bbox")
if not pb:
    print("    PLAT TIDAK TERDETEKSI -> masalah ada di detektor plat, bukan OCR.")
    sys.exit(0)

crop = A.crop_plate_with_padding(img, pb[0], pb[1], pb[2], pb[3])
cv2.imwrite(os.path.join(OUT, "plate_crop.jpg"), crop)
print(f"[3] crop plat shape={crop.shape} -> output/diag/plate_crop.jpg", flush=True)

print("[4] char model: kombinasi skala x imgsz x conf", flush=True)
cm = A.char_model
h, w = crop.shape[:2]
for scale_h in (0, 120, 200):
    if scale_h == 0:
        inp, tag = crop, "asli"
    else:
        s = scale_h / float(h)
        inp, tag = cv2.resize(crop, (max(8, int(w * s)), scale_h), interpolation=cv2.INTER_CUBIC), f"h={scale_h}"
    for imgsz in (160, 256, 384, 640):
        for conf in (0.10, 0.25):
            t = time.time()
            r = cm.predict(inp, imgsz=imgsz, conf=conf, verbose=False)[0]
            ms = (time.time() - t) * 1000
            items = []
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                items.append(((x1 + x2) / 2, (y1 + y2) / 2, r.names[int(b.cls[0])], float(b.conf[0])))
            items.sort(key=lambda z: z[0])
            txt = "".join(i[2] for i in items)
            confs = " ".join(f"{i[2]}:{i[3]:.2f}" for i in items)
            print(f"    skala={tag:7s} imgsz={imgsz:3d} conf={conf:.2f} {ms:5.0f}ms n={len(items):2d} '{txt}'  [{confs}]", flush=True)
            if imgsz == 160 and conf == 0.10:
                ov = inp.copy()
                for b in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(ov, r.names[int(b.cls[0])], (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.imwrite(os.path.join(OUT, f"chars_{tag}.jpg"), ov)

print("[5] RapidOCR pada crop yang sama", flush=True)
try:
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
    top = crop[0:max(8, int(h * 0.72)), :]
    t = time.time()
    rr, _ = eng(top, use_det=False, use_cls=False, use_rec=True)
    print(f"    rec-only (72% atas): {(time.time() - t) * 1000:.0f}ms -> {rr}", flush=True)
    t = time.time()
    rr, _ = eng(crop)
    print(f"    full det+rec: {(time.time() - t) * 1000:.0f}ms -> {[(x[1], round(float(x[2]), 2)) for x in (rr or [])]}", flush=True)
except Exception as e:
    print(f"    RapidOCR error: {e}", flush=True)

print("\nSELESAI. Kirim output ini + folder output/diag/ (plate_crop.jpg, chars_*.jpg).", flush=True)
