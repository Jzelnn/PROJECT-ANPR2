#!/usr/bin/env python3
"""
Uji latensi klasifikasi bodi 192 vs 224 (Langkah 3).

Mengukur waktu classify_vehicle_indonesian saja. Tidak menilai akurasi:
label 192 dan 224 hanya dibandingkan apakah stringnya sama.
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("SKIP_WARMUP", "1")
os.environ.setdefault("DISABLE_PADDLE", "1")

import cv2
import app as A


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (q / 100.0)
    low = int(pos)
    high = min(len(s) - 1, low + 1)
    w = pos - low
    return float(s[low] * (1.0 - w) + s[high] * w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    gt = os.path.join(ROOT, "tools", "eval", "ground_truth.csv")
    files = []
    with open(gt, encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            rel = line.split(",", 1)[0].strip()
            if not rel:
                continue
            path = os.path.join(ROOT, rel)
            if os.path.isfile(path):
                files.append(path)
            if len(files) >= args.limit:
                break

    sizes = (192, 224)
    samples = {sz: [] for sz in sizes}
    same = 0
    used = 0

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = [int(w * 0.2), int(h * 0.15), int(w * 0.8), int(h * 0.9)]
        labels = {}
        for _ in range(args.runs):
            order = sizes if used % 2 == 0 else (224, 192)
            for sz in order:
                os.environ["ANPR_BODY_IMGSZ"] = str(sz)
                t0 = time.perf_counter()
                _vt, style, _conf = A.classify_vehicle_indonesian(img, bbox, "car", 0.8)
                samples[sz].append((time.perf_counter() - t0) * 1000.0)
                labels[sz] = style
        used += 1
        if labels.get(192) == labels.get(224):
            same += 1
        print(f"[{used:02d}] 192={samples[192][-1]:6.1f}ms 224={samples[224][-1]:6.1f}ms label_same={labels.get(192)==labels.get(224)}", flush=True)

    os.environ.pop("ANPR_BODY_IMGSZ", None)
    print("=" * 72)
    print(f"BODY IMGSZ LATENCY | photos={used} | runs={args.runs} | label agreement {same}/{used}")
    print("Bukan ukuran akurasi. Tidak ada ground truth tipe bodi.")
    for sz in sizes:
        vals = samples[sz]
        print(f"  imgsz={sz}  p50={percentile(vals, 50):6.1f} ms  p95={percentile(vals, 95):6.1f} ms  n={len(vals)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
