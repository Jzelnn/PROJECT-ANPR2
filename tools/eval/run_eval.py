#!/usr/bin/env python3
"""
Skrip Evaluasi Terukur ANPR (Tahap 0 - Langkah 7)
=================================================
Menjalankan evaluasi pipeline ANPR pada daftar foto di tools/eval/ground_truth.csv
dan melaporkan:
1. Akurasi Exact-Match per kategori (jelas/buram/malam/miring/motor/dinas/tni) untuk baris yang sudah berlabel.
2. Latensi p50 & p95 per tahap (yolo_vehicle, plate_det, ocr, body_type, total_ms).
3. Rata-rata penggunaan CPU (CPU%) selama eksekusi menggunakan psutil (atau fallback process_time).
4. Jumlah pemanggilan fungsi OCR per foto.
5. Mendukung 2 mode:
   - --mode direct   : Memanggil run_anpr() langsung di dalam proses Python.
   - --mode endpoint : Mengirim HTTP POST ke /api/detect (mendukung header Authorization / X-API-Token).

Contoh penggunaan:
  python tools/eval/run_eval.py --gt tools/eval/ground_truth.csv --runs 5 --mode direct
  python tools/eval/run_eval.py --gt tools/eval/ground_truth.csv --runs 5 --mode endpoint --url http://127.0.0.1:5001/api/detect
"""

import argparse
import csv
import json
import os
import platform
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False


def normalize_plate(p):
    """Menormalisasi string plat nomor untuk perbandingan exact-match (hanya A-Z0-9, kecuali format militer dengan '-')."""
    if not p:
        return ""
    s = str(p).strip().upper()
    if s in ("-", "UNKNOWN", "TIDAK_TERBACA", "UNREADABLE", "NONE", "NULL"):
        return ""
    if "-" in s and all(part.isdigit() for part in s.split("-") if part):
        return s
    return re.sub(r"[^A-Z0-9]", "", s)


def percentile(values, q):
    """Menghitung persentil q (0..100) dari list angka."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (q / 100.0)
    low = int(pos)
    high = min(len(s) - 1, low + 1)
    weight = pos - low
    return float(s[low] * (1.0 - weight) + s[high] * weight)


def load_ground_truth(gt_path, limit=0):
    rows = []
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_file = (row.get("file") or "").strip()
            if not rel_file:
                continue
            abs_file = rel_file if os.path.isabs(rel_file) else os.path.join(ROOT, rel_file)
            rows.append({
                "file": rel_file,
                "abs_file": abs_file,
                "plat_benar": (row.get("plat_benar") or "").strip(),
                "kategori": (row.get("kategori") or "lainnya").strip().lower() or "lainnya",
                "catatan": (row.get("catatan") or "").strip(),
            })
            if limit and len(rows) >= limit:
                break
    return rows


class CpuSampler:
    """Mengukur CPU% proses selama blok evaluasi menggunakan psutil atau process_time."""
    def __init__(self):
        self.proc = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self.t0_wall = 0.0
        self.t0_cpu = 0.0

    def start(self):
        self.t0_wall = time.perf_counter()
        if self.proc is not None:
            try:
                ct = self.proc.cpu_times()
                self.t0_cpu = float(ct.user + ct.system)
                return
            except Exception:
                pass
        self.t0_cpu = time.process_time()

    def stop(self):
        dt_wall = max(1e-6, time.perf_counter() - self.t0_wall)
        if self.proc is not None:
            try:
                ct = self.proc.cpu_times()
                dt_cpu = max(0.0, float(ct.user + ct.system) - self.t0_cpu)
                return (dt_cpu / dt_wall) * 100.0
            except Exception:
                pass
        dt_cpu = max(0.0, time.process_time() - self.t0_cpu)
        return (dt_cpu / dt_wall) * 100.0


def extract_stage_timings(res_dict):
    """Mengekstrak waktu per tahap (ms) dari respons run_anpr / /api/detect."""
    perf = res_dict.get("perf_breakdown") or {}
    stages = res_dict.get("_debug_timing_stages") or []

    yolo_ms = float(perf.get("yolo_ms", 0.0) or 0.0)
    plate_ms = float(perf.get("plate_ms", 0.0) or 0.0)
    ocr_ms = float(perf.get("ocr_ms", 0.0) or 0.0)
    body_ms = float(perf.get("body_ms", 0.0) or 0.0)
    total_ms = float(perf.get("total_ms", 0.0) or 0.0)

    if stages:
        p_sum = sum(float(s.get("ms", 0.0)) for s in stages if str(s.get("stage", "")).startswith("plate_"))
        if p_sum > 0 and plate_ms == 0.0:
            plate_ms = p_sum
        b_sum = sum(float(s.get("ms", 0.0)) for s in stages if s.get("stage") == "body_type")
        if b_sum > 0 and body_ms == 0.0:
            body_ms = b_sum
        o_sum = sum(float(s.get("ms", 0.0)) for s in stages if s.get("stage") == "char_ocr_and_validate")
        if o_sum > 0 and ocr_ms == 0.0:
            ocr_ms = o_sum

    if total_ms == 0.0 and res_dict.get("detection_time_sec") is not None:
        total_ms = float(res_dict["detection_time_sec"]) * 1000.0

    return {
        "yolo_ms": yolo_ms,
        "plate_ms": plate_ms,
        "ocr_ms": ocr_ms,
        "body_ms": body_ms,
        "total_ms": total_ms,
    }


def _default_gt_path():
    local_gt = os.path.join(ROOT, "tools", "eval", "ground_truth.local.csv")
    if os.path.isfile(local_gt):
        return local_gt
    return os.path.join(ROOT, "tools", "eval", "ground_truth.csv")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Terukur Pipeline ANPR")
    parser.add_argument("--gt", default=_default_gt_path(), help="Path file ground_truth.local.csv atau ground_truth.csv")
    parser.add_argument("--mode", choices=["direct", "endpoint"], default="direct", help="Mode evaluasi: direct (run_anpr) atau endpoint (POST /api/detect)")
    parser.add_argument("--url", default="http://127.0.0.1:5001/api/detect", help="URL endpoint untuk --mode endpoint")
    parser.add_argument("--token", default=os.environ.get("ANPR_API_TOKEN", ""), help="API token untuk --mode endpoint")
    parser.add_argument("--runs", type=int, default=5, help="Jumlah pengulangan per foto (default: 5)")
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah foto dari CSV (0 = semua)")
    parser.add_argument("--out", default="", help="Simpan hasil JSON ke file tertentu")
    args = parser.parse_args()

    os.environ["ANPR_DEBUG_TIMING"] = "1"
    os.environ.setdefault("SKIP_WARMUP", "1")

    if not os.path.isfile(args.gt):
        print(f"[ERROR] File ground truth tidak ditemukan: {args.gt}")
        sys.exit(1)

    gt_rows = load_ground_truth(args.gt, limit=args.limit)
    valid_rows = [r for r in gt_rows if os.path.isfile(r["abs_file"])]
    missing_rows = [r for r in gt_rows if not os.path.isfile(r["abs_file"])]

    print("=" * 88)
    print(f"EVALUASI ANPR | Mode: {args.mode.upper()} | Runs: {args.runs}x | Foto Valid: {len(valid_rows)}/{len(gt_rows)} | Arch: {platform.machine()}")
    print("=" * 88, flush=True)
    if missing_rows:
        print(f"[WARN] {len(missing_rows)} file gambar di CSV tidak ditemukan di disk.")

    app_mod = None
    cv2_mod = None
    ocr_call_counter = {"count": 0}

    if args.mode == "direct":
        import cv2 as cv2_mod
        import app as app_mod

        # Bungkus fungsi OCR untuk menghitung jumlah panggilan OCR per foto secara akurat
        orig_char_fn = getattr(app_mod, "read_plate_with_char_model", None)
        orig_rapid_fn = getattr(app_mod, "rapidocr_read_plate", None)
        orig_paddle_fn = getattr(app_mod, "read_plate_with_paddleocr", None)

        if orig_char_fn:
            def _wrapped_char(*a, **kw):
                ocr_call_counter["count"] += 1
                return orig_char_fn(*a, **kw)
            app_mod.read_plate_with_char_model = _wrapped_char

        if orig_rapid_fn:
            def _wrapped_rapid(*a, **kw):
                ocr_call_counter["count"] += 1
                return orig_rapid_fn(*a, **kw)
            app_mod.rapidocr_read_plate = _wrapped_rapid

        if orig_paddle_fn:
            def _wrapped_paddle(*a, **kw):
                ocr_call_counter["count"] += 1
                return orig_paddle_fn(*a, **kw)
            app_mod.read_plate_with_paddleocr = _wrapped_paddle
    else:
        import requests

    stage_samples = {
        "yolo_ms": [],
        "plate_ms": [],
        "ocr_ms": [],
        "body_ms": [],
        "total_ms": [],
        "wall_ms": [],
        "cpu_pct": [],
        "ocr_calls": [],
    }

    per_image_results = []
    sampler = CpuSampler()

    for idx, row in enumerate(valid_rows, start=1):
        img_runs_stages = []
        img_runs_wall = []
        img_runs_cpu = []
        img_runs_ocr_calls = []
        last_res = {}

        if args.mode == "direct":
            img_bgr = cv2_mod.imread(row["abs_file"])
            if img_bgr is None:
                print(f"[{idx:02d}/{len(valid_rows):02d}] GAGAL BACA: {row['file']}")
                continue

            for r_i in range(args.runs):
                app_mod.confirmation_manager.tracks.clear()
                ocr_call_counter["count"] = 0
                sampler.start()
                t_w0 = time.perf_counter()
                res = app_mod.run_anpr(
                    img_bgr.copy(),
                    is_stream=False,
                    frame_id=(idx * 100 + r_i),
                    single_vehicle_mode=True,
                )
                wall_ms = (time.perf_counter() - t_w0) * 1000.0
                cpu_p = sampler.stop()
                st = extract_stage_timings(res)
                img_runs_stages.append(st)
                img_runs_wall.append(wall_ms)
                img_runs_cpu.append(cpu_p)
                img_runs_ocr_calls.append(ocr_call_counter["count"])
                last_res = res
        else:
            with open(row["abs_file"], "rb") as f_in:
                img_bytes = f_in.read()
            headers = {}
            if args.token:
                headers["Authorization"] = f"Bearer {args.token}"
                headers["X-API-Token"] = args.token
            for r_i in range(args.runs):
                sampler.start()
                t_w0 = time.perf_counter()
                resp = requests.post(
                    args.url,
                    files={"image": (os.path.basename(row["abs_file"]), img_bytes, "image/jpeg")},
                    headers=headers,
                    timeout=30,
                )
                wall_ms = (time.perf_counter() - t_w0) * 1000.0
                cpu_p = sampler.stop()
                res = resp.json() if resp.status_code == 200 else {}
                st = extract_stage_timings(res)
                ocr_c = int(res.get("ocr_calls_count", 1 if st["ocr_ms"] > 0 else 0))
                img_runs_stages.append(st)
                img_runs_wall.append(wall_ms)
                img_runs_cpu.append(cpu_p)
                img_runs_ocr_calls.append(ocr_c)
                last_res = res

        # Ambil median (p50) dari N runs untuk foto ini
        med_yolo = statistics.median(s["yolo_ms"] for s in img_runs_stages)
        med_plate = statistics.median(s["plate_ms"] for s in img_runs_stages)
        med_ocr = statistics.median(s["ocr_ms"] for s in img_runs_stages)
        med_body = statistics.median(s["body_ms"] for s in img_runs_stages)
        med_total = statistics.median(s["total_ms"] for s in img_runs_stages)
        med_wall = statistics.median(img_runs_wall)
        med_cpu = statistics.median(img_runs_cpu)
        med_ocr_calls = statistics.median(img_runs_ocr_calls)

        stage_samples["yolo_ms"].append(med_yolo)
        stage_samples["plate_ms"].append(med_plate)
        stage_samples["ocr_ms"].append(med_ocr)
        stage_samples["body_ms"].append(med_body)
        stage_samples["total_ms"].append(med_total)
        stage_samples["wall_ms"].append(med_wall)
        stage_samples["cpu_pct"].append(med_cpu)
        stage_samples["ocr_calls"].append(med_ocr_calls)

        dets = last_res.get("detections") or []
        d0 = dets[0] if dets else {}
        pred_plate = d0.get("license_plate") or ""
        pred_status = d0.get("status") or ("NO_VEHICLE" if not dets else "UNKNOWN")
        pred_vtype = d0.get("vehicle_type") or "-"
        pred_body = d0.get("body_style") or "-"
        pred_conf = float(d0.get("plate_confidence") or 0.0)

        gt_norm = normalize_plate(row["plat_benar"])
        pred_norm = normalize_plate(pred_plate)
        is_labeled = bool(gt_norm)
        is_match = (pred_norm == gt_norm) if is_labeled else None

        match_tag = "UNLABELED" if not is_labeled else ("MATCH" if is_match else f"MISMATCH(gt={row['plat_benar']})")
        print(
            f"[{idx:02d}/{len(valid_rows):02d}] {row['kategori']:7s} | total={med_total:6.1f}ms (Y={med_yolo:5.1f} P={med_plate:5.1f} O={med_ocr:5.1f} B={med_body:5.1f}) "
            f"| ocr_calls={med_ocr_calls:.0f} | pred={pred_plate!r:12s} ({pred_conf:.2f}) | {match_tag} | {row['file']}",
            flush=True,
        )

        per_image_results.append({
            "file": row["file"],
            "kategori": row["kategori"],
            "plat_benar": row["plat_benar"],
            "pred_plate": pred_plate,
            "pred_status": pred_status,
            "vehicle_type": pred_vtype,
            "body_style": pred_body,
            "plate_confidence": round(pred_conf, 3),
            "labeled": is_labeled,
            "exact_match": is_match,
            "yolo_ms_p50": round(med_yolo, 2),
            "plate_ms_p50": round(med_plate, 2),
            "ocr_ms_p50": round(med_ocr, 2),
            "body_ms_p50": round(med_body, 2),
            "total_ms_p50": round(med_total, 2),
            "wall_ms_p50": round(med_wall, 2),
            "cpu_pct_p50": round(med_cpu, 1),
            "ocr_calls_p50": round(med_ocr_calls, 1),
        })

    # Ringkasan Latensi p50 / p95
    print("\n" + "=" * 88)
    print("RINGKASAN LATENSI PER TAHAP (ms) & BEBAN CPU")
    print("=" * 88)
    summary_latency = {}
    for k in ("yolo_ms", "plate_ms", "ocr_ms", "body_ms", "total_ms", "wall_ms"):
        vals = stage_samples[k]
        p50_v = round(percentile(vals, 50), 2)
        p95_v = round(percentile(vals, 95), 2)
        summary_latency[k] = {"p50": p50_v, "p95": p95_v}
        print(f"  - {k:12s} : p50 = {p50_v:7.2f} ms | p95 = {p95_v:7.2f} ms")

    avg_cpu = round(statistics.mean(stage_samples["cpu_pct"]), 1) if stage_samples["cpu_pct"] else 0.0
    avg_ocr_calls = round(statistics.mean(stage_samples["ocr_calls"]), 2) if stage_samples["ocr_calls"] else 0.0
    print(f"  - Rata-rata CPU% proses       : {avg_cpu:.1f}%")
    print(f"  - Rata-rata panggilan OCR/foto: {avg_ocr_calls:.2f}x")

    # Ringkasan Akurasi per Kategori
    print("\n" + "=" * 88)
    print("RINGKASAN AKURASI EXACT-MATCH PER KATEGORI")
    print("=" * 88)
    categories = sorted(set(r["kategori"] for r in per_image_results))
    cat_summary = {}
    total_labeled = 0
    total_matched = 0
    for cat in categories:
        cat_rows = [r for r in per_image_results if r["kategori"] == cat]
        labeled_rows = [r for r in cat_rows if r["labeled"]]
        matched_rows = [r for r in labeled_rows if r["exact_match"]]
        total_labeled += len(labeled_rows)
        total_matched += len(matched_rows)
        acc = (len(matched_rows) / len(labeled_rows) * 100.0) if labeled_rows else None
        cat_summary[cat] = {
            "total_photos": len(cat_rows),
            "labeled_photos": len(labeled_rows),
            "exact_matches": len(matched_rows),
            "accuracy_pct": round(acc, 1) if acc is not None else None,
        }
        acc_str = f"{acc:5.1f}% ({len(matched_rows)}/{len(labeled_rows)})" if acc is not None else f"BELUM BERLABEL (0/{len(cat_rows)} diisi)"
        print(f"  - Kategori {cat:10s} : {acc_str} [Total foto: {len(cat_rows)}]")

    overall_acc = round(total_matched / total_labeled * 100.0, 1) if total_labeled > 0 else None
    print("-" * 88)
    if overall_acc is not None:
        print(f"  TOTAL AKURASI (BERLABEL): {overall_acc:.1f}% ({total_matched}/{total_labeled} foto berlabel, {len(per_image_results) - total_labeled} belum diisi)")
    else:
        print(f"  TOTAL AKURASI: Belum ada plat_benar yang diisi di {args.gt} ({len(per_image_results)} foto dievaluasi).")
    print("=" * 88, flush=True)

    out_dir = os.path.join(ROOT, "output", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_file = args.out or os.path.join(out_dir, f"eval_{platform.machine().lower()}_{int(time.time())}.json")
    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.machine(),
        "platform": sys.platform,
        "mode": args.mode,
        "runs": args.runs,
        "total_photos": len(per_image_results),
        "labeled_photos": total_labeled,
        "exact_matches": total_matched,
        "overall_accuracy_pct": overall_acc,
        "avg_cpu_pct": avg_cpu,
        "avg_ocr_calls_per_photo": avg_ocr_calls,
        "latency_ms": summary_latency,
        "categories": cat_summary,
        "results": per_image_results,
    }
    with open(out_file, "w", encoding="utf-8") as f_out:
        json.dump(report_payload, f_out, indent=2, ensure_ascii=False)
    print(f"[INFO] Laporan JSON disimpan ke: {out_file}", flush=True)


if __name__ == "__main__":
    main()
