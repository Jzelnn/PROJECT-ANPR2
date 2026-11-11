#!/usr/bin/env python3
"""
Bandingkan bacaan OCR lama vs aturan Langkah 4 pada foto yang sama.
Lama: model karakter melihat crop penuh, string lebih panjang boleh menang.
Uji: ANPR_CHAR_ROW_RATIO=0.74. Default produksi tetap 1 karena 0.74 memotong digit.
Tanpa plat_benar ini hanya melaporkan string yang berubah, bukan akurasi.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
os.environ.setdefault("SKIP_WARMUP", "1")
os.environ.setdefault("DISABLE_PADDLE", "1")

import cv2
import app as A


def _plate(img):
    A.confirmation_manager.reset()
    res = A.run_anpr(img, is_stream=False, single_vehicle_mode=True)
    dets = res.get("detections") or []
    if not dets:
        return "", 0.0, ""
    d = dets[0]
    return (d.get("license_plate") or ""), float(d.get("ocr_confidence") or 0.0), d.get("status") or ""


def main():
    limit = int(os.environ.get("ANPR_COMPARE_LIMIT", "30"))
    gt = os.path.join(ROOT, "tools", "eval", "ground_truth.csv")
    files = []
    with open(gt, encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            rel = line.split(",", 1)[0].strip()
            path = os.path.join(ROOT, rel) if rel else ""
            if path and os.path.isfile(path):
                files.append((rel, path))
            if len(files) >= limit:
                break

    changed = 0
    for rel, path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        os.environ["ANPR_CHAR_ROW_RATIO"] = "1"
        os.environ["ANPR_OCR_LONGER_WINS"] = "1"
        old, old_c, old_s = _plate(img)
        os.environ["ANPR_CHAR_ROW_RATIO"] = "0.74"
        os.environ["ANPR_OCR_LONGER_WINS"] = "0"
        new, new_c, new_s = _plate(img)
        same = A._plate_norm(old) == A._plate_norm(new)
        if not same:
            changed += 1
        print(f"{'SAME' if same else 'DIFF'} | old={old!r:16} {old_s:12} | new={new!r:16} {new_s:12} | {rel}", flush=True)

    os.environ.pop("ANPR_CHAR_ROW_RATIO", None)
    os.environ.pop("ANPR_OCR_LONGER_WINS", None)
    print(f"BERUBAH {changed}/{len(files)} (bukan skor akurasi; plat_benar masih kosong)")


if __name__ == "__main__":
    main()
