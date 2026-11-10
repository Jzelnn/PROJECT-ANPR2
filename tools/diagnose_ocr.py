"""
Investigasi & Instrumentasi TAHAP A (Timing 2,1 Detik) & TAHAP B (Detektor Plat) + Diagnosa OCR.

Pakai di Raspberry Pi / Laptop:
    SKIP_WARMUP=1 DISABLE_PADDLE=1 ANPR_DEBUG_TIMING=1 python tools/diagnose_ocr.py [path/foto_mobil.jpg]
"""
import os
import sys
import time
import json
import glob
import statistics

os.environ.setdefault("SKIP_WARMUP", "1")
os.environ.setdefault("DISABLE_PADDLE", "1")
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ["ANPR_DEBUG_TIMING"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.chdir(ROOT)

import cv2
import numpy as np
import torch

# Cari foto uji otomatis jika tidak diberikan di argumen
CANDIDATE_IMGS = [
    os.path.join(ROOT, "output", "results", "temp_upload.jpg"),
    os.path.join(ROOT, "output", "captures", "latest_result.jpg"),
    os.path.join(ROOT, "output", "screenshots", "latest_result.jpg"),
]
for g in sorted(glob.glob(os.path.join(ROOT, "output", "**", "*.jpg"), recursive=True)):
    if "diag" not in g and "debug_crops" not in g and "debug_plates" not in g:
        CANDIDATE_IMGS.append(g)

img_path = sys.argv[1] if len(sys.argv) >= 2 else next((p for p in CANDIDATE_IMGS if os.path.exists(p)), None)
if not img_path or not os.path.exists(img_path):
    print("File foto uji tidak ditemukan. Pakai: python tools/diagnose_ocr.py path/foto_mobil.jpg")
    sys.exit(1)

print("[0] Memuat backend/app.py...", flush=True)
t0 = time.perf_counter()
import app as A
print(f"    selesai dalam {(time.perf_counter() - t0):.2f}s\n", flush=True)

OUT = os.path.join(ROOT, "output", "diag")
os.makedirs(OUT, exist_ok=True)

with open(img_path, "rb") as f:
    raw_file_bytes = f.read()

img_orig = cv2.imread(img_path)
if img_orig is None:
    print(f"Gagal membaca gambar: {img_path}")
    sys.exit(1)
ih, iw = img_orig.shape[:2]
cv2.imwrite(os.path.join(OUT, "orig_full.jpg"), img_orig)
print(f"[INFO] Foto uji: {img_path} (shape={img_orig.shape}, bytes={len(raw_file_bytes)})")
print(f"       Disimpan ke: output/diag/orig_full.jpg\n", flush=True)

# ==============================================================================
# CEK MODEL METADATA & SOFTMAX KLASIFIKASI
# ==============================================================================
print("=" * 80)
print("CEK METADATA MODEL & SOFTMAX BODY_TYPE")
print("=" * 80)
for m_label, m_obj in [
    ("vehicle_model (models/vehicle/best.pt)", A.vehicle_model),
    ("plate_detector (models/plate/best.pt)", A.plate_detector),
    ("plate_model_legacy (models/plate/legacy.pt)", A.plate_model_legacy),
    ("body_style_model (models/body_type/best.pt)", A.body_style_model),
    ("char_model (models/character/best.pt)", A.char_model),
]:
    names = getattr(m_obj, "names", {})
    print(f"  - {m_label:45s}: nc={len(names):2d} names={names}")

try:
    bm = A.body_style_model.model
    bm.eval()
    dummy_t = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.no_grad():
        b_out = bm(dummy_t)
    b_pred = b_out[0] if isinstance(b_out, (list, tuple)) else b_out
    print(f"  - Cek body_style_model raw output: type(out)={type(b_out).__name__} pred.shape={tuple(b_pred.shape)} "
          f"min={float(b_pred[0].min()):.4f} sum={float(b_pred[0].sum(-1)):.4f} "
          f"top1_raw={float(b_pred[0].max()):.4f} vs top1_double_softmax={float(b_pred[0].softmax(dim=-1).max()):.4f}\n", flush=True)
except Exception as e:
    print(f"  - Cek body_style_model error: {e}\n", flush=True)

# ==============================================================================
# TAHAP A: INVESTIGASI WAKTU 5x RUN (/api/detect END-TO-END)
# ==============================================================================
print("=" * 80)
print("TAHAP A: INVESTIGASI WAKTU (5x RUN PADA FOTO YANG SAMA)")
print("=" * 80)

N_RUNS = 5
stage_runs = {}
stage_meta = {}
stage_order = []
perf_totals_pre_ann = []
perf_totals_endpoint = []
logged_perf_yolo = []
logged_perf_track = []
logged_perf_body = []
logged_perf_ocr = []
logged_perf_total = []

last_res = None
for run_idx in range(N_RUNS):
    t_ep0 = time.perf_counter()

    # 1. Decode gambar (persis /api/detect baris 5600-5602)
    t_d0 = time.perf_counter()
    nparr = np.frombuffer(raw_file_bytes, np.uint8)
    img_dec = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    dec_ms = (time.perf_counter() - t_d0) * 1000.0

    # 2. Panggil run_anpr(img, is_stream=False)
    res = A.run_anpr(img_dec, is_stream=False)
    last_res = res

    # 3. Simpan record & serialisasi JSON (persis /api/detect baris 5615-5626)
    t_s0 = time.perf_counter()
    dets = res.get("detections") or []
    if dets:
        p_det = dets[0]
        if p_det.get("is_newly_confirmed"):
            A.save_parking_record(p_det, source_img=img_dec)
    sav_ms = (time.perf_counter() - t_s0) * 1000.0

    t_j0 = time.perf_counter()
    _ = json.dumps({k: v for k, v in res.items() if not k.startswith("_")})
    json_ms = (time.perf_counter() - t_j0) * 1000.0

    ep_ms = (time.perf_counter() - t_ep0) * 1000.0
    perf_totals_endpoint.append(ep_ms)

    pb_info = res.get("perf_breakdown") or {}
    logged_perf_yolo.append(float(pb_info.get("yolo_ms", 0.0)))
    logged_perf_track.append(float(pb_info.get("track_ms", 0.0)))
    logged_perf_body.append(float(pb_info.get("body_ms", 0.0)))
    logged_perf_ocr.append(float(pb_info.get("ocr_ms", 0.0)))
    logged_perf_total.append(float(pb_info.get("total_ms", 0.0)))

    dbg_sum = res.get("_debug_timing_summary") or {}
    perf_totals_pre_ann.append(float(dbg_sum.get("total_pre_annotate_ms", pb_info.get("total_ms", 0.0))))

    all_stages_this_run = [
        {"stage": "0_decode_imdecode", "model": "cv2.imdecode", "imgsz": f"{iw}x{ih}", "ms": dec_ms, "det_count": 1, "raw_count": 1}
    ] + (res.get("_debug_timing_stages") or []) + [
        {"stage": "98_save_parking_record", "model": "-", "imgsz": "-", "ms": sav_ms, "det_count": len(dets), "raw_count": len(dets)},
        {"stage": "99_json_serialize", "model": "json.dumps", "imgsz": "-", "ms": json_ms, "det_count": 1, "raw_count": 1},
    ]

    for st in all_stages_this_run:
        sname = st["stage"]
        if sname not in stage_runs:
            stage_runs[sname] = []
            stage_order.append(sname)
            stage_meta[sname] = (st["model"], st["imgsz"], st.get("raw_count", 0), st.get("det_count", 0))
        stage_runs[sname].append(float(st["ms"]))

print("\n--- TABEL WAKTU PER TAHAP (p50 / MEDIAN DARI 5x RUN) ---")
print(f"{'Tahap':<30} | {'Model':<28} | {'imgsz':<11} | {'ms (p50)':>9} | {'Deteksi (raw -> valid)'}")
print("-" * 108)
sum_p50_inside_run_anpr_pre_ann = 0.0
sum_p50_all_endpoint = 0.0
for sname in stage_order:
    p50_ms = statistics.median(stage_runs[sname])
    mdl, sz, raw_c, val_c = stage_meta[sname]
    print(f"{sname:<30} | {str(mdl):<28} | {str(sz):<11} | {p50_ms:9.2f} | raw={raw_c} -> valid={val_c}")
    sum_p50_all_endpoint += p50_ms
    if sname not in ("0_decode_imdecode", "encode_and_save_result", "98_save_parking_record", "99_json_serialize"):
        sum_p50_inside_run_anpr_pre_ann += p50_ms

p50_perf_total = statistics.median(perf_totals_pre_ann)
p50_ep_total = statistics.median(perf_totals_endpoint)
p50_log_yolo = statistics.median(logged_perf_yolo)
p50_log_track = statistics.median(logged_perf_track)
p50_log_body = statistics.median(logged_perf_body)
p50_log_ocr = statistics.median(logged_perf_ocr)
sum_logged_perf = p50_log_yolo + p50_log_track + p50_log_body + p50_log_ocr

print("-" * 108)
print(f"1. Yang tercatat di [PERF] lama (YOLO + Tracking + Body + OCR) : {sum_logged_perf:8.2f} ms "
      f"(YOLO={p50_log_yolo:.1f}, Track={p50_log_track:.1f}, Body={p50_log_body:.1f}, OCR={p50_log_ocr:.1f})")
print(f"2. Total [PERF] run_anpr (sebelum encode gambar)               : {p50_perf_total:8.2f} ms")
print(f"   -> Selisih tak tercatat di [PERF] lama                      : {(p50_perf_total - sum_logged_perf):8.2f} ms "
      f"(ini berasal dari plate_det_crop + plate_legacy_crop + plate_global_full + body_type yang tidak dimasukkan ke variabel t_body_total/t_ocr_total!)")
print(f"3. Jumlah seluruh tahap terinstrumentasi (pre-annotate)        : {sum_p50_inside_run_anpr_pre_ann:8.2f} ms")
print(f"   -> Selisih terhadap Total run_anpr (unexplained pre-ann)    : {(p50_perf_total - sum_p50_inside_run_anpr_pre_ann):8.2f} ms")
print(f"4. Jumlah seluruh tahap Endpoint (/api/detect lengkap)         : {sum_p50_all_endpoint:8.2f} ms")
print(f"5. Total Endpoint terukur (p50 dari 5x)                        : {p50_ep_total:8.2f} ms")
print(f"   -> Selisih terhadap Total Endpoint (unexplained endpoint)   : {(p50_ep_total - sum_p50_all_endpoint):8.2f} ms\n", flush=True)

# ==============================================================================
# TAHAP B: KENAPA PLAT TIDAK TERDETEKSI (BUMPER CROP, OVERLAY, TES DIFERENSIAL)
# ==============================================================================
print("=" * 80)
print("TAHAP B: INVESTIGASI DETEKTOR PLAT (CROP, FILTER GEOMETRI, & TES SKALA/CONF)")
print("=" * 80)

dets = (last_res or {}).get("detections") or []
if not dets:
    print("TIDAK ADA KENDARAAN TERDETEKSI pada foto ini.")
    sys.exit(0)

d0 = dets[0]
vx1, vy1, vx2, vy2 = d0["bbox"]
vw = max(1, vx2 - vx1)
vh = max(1, vy2 - vy1)
pad_x = int(vw * 0.06)
pad_y_top = int(vh * 0.04)
pad_y_bot = int(vh * 0.10)
cx1 = max(0, vx1 - pad_x)
cy1 = max(0, vy1 - pad_y_top)
cx2 = min(iw, vx2 + pad_x)
cy2 = min(ih, vy2 + pad_y_bot)
bumper_crop = img_orig[cy1:cy2, cx1:cx2].copy()

cv2.imwrite(os.path.join(OUT, "bumper_crop.jpg"), bumper_crop)
ov_full_coords = img_orig.copy()
cv2.rectangle(ov_full_coords, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
cv2.rectangle(ov_full_coords, (cx1, cy1), (cx2, cy2), (255, 200, 0), 2)
cv2.putText(ov_full_coords, f"veh={vx1},{vy1},{vx2},{vy2} padded={cx1},{cy1},{cx2},{cy2}",
            (max(10, cx1), max(25, cy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
cv2.imwrite(os.path.join(OUT, "orig_with_vehicle_and_bumper_coords.jpg"), ov_full_coords)

print(f"[B1] Koordinat Kendaraan & Bumper Crop:")
print(f"     - Image shape  : {iw}x{ih} (W x H)")
print(f"     - Vehicle bbox : [{vx1}, {vy1}, {vx2}, {vy2}] (vw={vw}, vh={vh}, type={d0.get('vehicle_type')}, body={d0.get('body_style')})")
print(f"     - Bumper crop  : [{cx1}, {cy1}, {cx2}, {cy2}] (shape={bumper_crop.shape[1]}x{bumper_crop.shape[0]}) -> output/diag/bumper_crop.jpg")
print(f"     - Pipeline plate_bbox hasil akhir: {d0.get('plate_bbox')} | license_plate={d0.get('license_plate')!r}\n", flush=True)


recovered_plate_boxes = []


def inspect_raw_yolo_and_filters(model_obj, model_tag, inp_img, is_crop, imgsz_list, conf_list):
    """
    Menjalankan model detektor plat dan menginspeksi:
    1) Skor maksimum mentah per kelas langsung dari tensor output model (sebelum filter conf).
    2) Cek apakah argmax lintas kelas (mobil vs plat-nomor) menekan skor plat-nomor.
    3) Deteksi pada conf (0.25, 0.10, 0.05) beserta alasan mengapa box diterima/ditolak oleh filter pascaproses.
    """
    h0, w0 = inp_img.shape[:2]
    names = getattr(model_obj, "names", {})
    m_inner = model_obj.model
    m_inner.eval()

    print(f"--- Model: {model_tag} | Input: {'bumper_crop' if is_crop else 'full_img'} ({w0}x{h0}) | Classes: {names} ---")
    for sz in imgsz_list:
        r = min(sz / max(1, h0), sz / max(1, w0))
        new_unpad = (max(1, int(round(w0 * r))), max(1, int(round(h0 * r))))
        pad_w = (32 - (new_unpad[0] % 32)) % 32
        pad_h = (32 - (new_unpad[1] % 32)) % 32
        dw, dh = pad_w / 2.0, pad_h / 2.0
        interp = cv2.INTER_CUBIC if r > 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(inp_img, new_unpad, interpolation=interp)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(blob).unsqueeze(0)

        t_m0 = time.perf_counter()
        with torch.no_grad():
            out = m_inner(tensor)
        infer_ms = (time.perf_counter() - t_m0) * 1000.0
        pred = (out[0] if isinstance(out, (list, tuple)) else out).transpose(1, 2)[0]
        scores_all = pred[:, 4:]
        per_class_max = {
            f"{cid}:{names.get(cid, str(cid))}": round(float(scores_all[:, cid].max()), 4)
            for cid in range(scores_all.shape[1])
        }
        # Cek anchor terbaik khusus kelas plat (cls=2 pada best.pt, cls=0 pada legacy.pt)
        plate_cid = 2 if scores_all.shape[1] > 2 else 0
        best_p_idx = int(scores_all[:, plate_cid].argmax())
        best_p_score = float(scores_all[best_p_idx, plate_cid])
        best_p_cxcywh = pred[best_p_idx, :4].tolist()
        px1 = int(round(max(0, min(w0, (best_p_cxcywh[0] - best_p_cxcywh[2] / 2 - left) / r))))
        py1 = int(round(max(0, min(h0, (best_p_cxcywh[1] - best_p_cxcywh[3] / 2 - top) / r))))
        px2 = int(round(max(0, min(w0, (best_p_cxcywh[0] + best_p_cxcywh[2] / 2 - left) / r))))
        py2 = int(round(max(0, min(h0, (best_p_cxcywh[1] + best_p_cxcywh[3] / 2 - top) / r))))
        abs_best_p = [cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2] if is_crop else [px1, py1, px2, py2]
        suppressed_by_argmax = bool(scores_all.shape[1] > 1 and int(scores_all[best_p_idx].argmax()) != plate_cid)
        print(f"  [imgsz={sz:3d} -> padded={padded.shape[1]}x{padded.shape[0]} | {infer_ms:5.1f}ms] Max raw score per kelas: {per_class_max}")
        print(f"      Anchor plat tertinggi (cls={plate_cid}): score={best_p_score:.4f} abs_box={abs_best_p} suppressed_by_other_cls={suppressed_by_argmax}")
        if best_p_score >= 0.04 and (abs_best_p[2] - abs_best_p[0]) >= 12 and (abs_best_p[3] - abs_best_p[1]) >= 6:
            recovered_plate_boxes.append((best_p_score, abs_best_p, f"{model_tag}_sz{sz}"))

        for cf in conf_list:
            res = model_obj.predict(inp_img, conf=cf, imgsz=sz, verbose=False)[0]
            box_summaries = []
            for b in res.boxes:
                cid = int(b.cls[0])
                cname = names.get(cid, str(cid))
                bconf = float(b.conf[0])
                bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                abs_b = [cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2] if is_crop else [bx1, by1, bx2, by2]

                # Cek alasan filter di run_anpr
                reasons = []
                if cname not in ["plat-nomor", "license_plate"] and len(names) > 1:
                    reasons.append(f"REJECT_CLASS({cname})")
                if not A.is_valid_plate_box(abs_b, iw, ih):
                    bw_a, bh_a = max(1, abs_b[2] - abs_b[0]), max(1, abs_b[3] - abs_b[1])
                    reasons.append(f"REJECT_GEOM(w={bw_a},h={bh_a},ar={bw_a/bh_a:.2f})")
                rel_y = ((abs_b[1] + abs_b[3]) / 2.0 - vy1) / float(vh)
                if d0.get("vehicle_type") != "motorcycle" and rel_y < 0.25:
                    reasons.append(f"REJECT_REL_Y({rel_y:.2f}<0.25)")
                if not is_crop:
                    pcx, pcy = (abs_b[0] + abs_b[2]) / 2.0, (abs_b[1] + abs_b[3]) / 2.0
                    if not ((vx1 - 35 <= pcx <= vx2 + 35) and (vy1 - 20 <= pcy <= vy2 + 45)):
                        reasons.append("REJECT_OUTSIDE_VEH_MARGIN")

                status_str = "ACCEPTED" if not reasons else ",".join(reasons)
                box_summaries.append(f"{cname}({bconf:.3f}) box={abs_b} rel_y={rel_y:.2f} -> {status_str}")
            print(f"      conf={cf:.2f}: n={len(res.boxes)} | {'; '.join(box_summaries) if box_summaries else 'TIDAK ADA DETEKSI'}")

            # Simpan overlay pada conf=0.05
            if cf == 0.05 and sz in (256, 384, 640):
                ov = inp_img.copy()
                for b in res.boxes:
                    cid = int(b.cls[0])
                    cname = names.get(cid, str(cid))
                    bconf = float(b.conf[0])
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                    color = (0, 255, 0) if cname in ("plat-nomor", "license_plate") else (0, 165, 255)
                    cv2.rectangle(ov, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(ov, f"{cname} {bconf:.2f}", (bx1, max(15, by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                tag_clean = model_tag.replace("/", "_").replace(" ", "_")
                inp_clean = "crop" if is_crop else "full"
                cv2.imwrite(os.path.join(OUT, f"overlay_{tag_clean}_{inp_clean}_sz{sz}.jpg"), ov)
    print("", flush=True)


inspect_raw_yolo_and_filters(A.plate_detector, "plate_best", bumper_crop, True, (256, 384, 640), (0.25, 0.10, 0.05))
inspect_raw_yolo_and_filters(A.plate_model_legacy, "plate_legacy", bumper_crop, True, (256, 384, 640, 960), (0.25, 0.10, 0.05))
inspect_raw_yolo_and_filters(A.plate_detector, "plate_best", img_orig, False, (384, 640), (0.25, 0.10, 0.05))

# Jika ada plat yang berhasil ditemukan pada salah satu imgsz, jalankan juga sapuan OCR char_model
pb_final = d0.get("plate_bbox")
if not pb_final and recovered_plate_boxes:
    recovered_plate_boxes.sort(key=lambda x: -x[0])
    pb_final = recovered_plate_boxes[0][1]
    print(f"[INFO] Menggunakan plate_bbox terbaik dari sapuan imgsz ({recovered_plate_boxes[0][2]}, score={recovered_plate_boxes[0][0]:.3f}): {pb_final}")

if pb_final:
    crop_p = A.crop_plate_with_padding(img_orig, pb_final[0], pb_final[1], pb_final[2], pb_final[3])
    cv2.imwrite(os.path.join(OUT, "plate_crop.jpg"), crop_p)
    print(f"\n[SAPUAN OCR PADA PLATE CROP ({crop_p.shape[1]}x{crop_p.shape[0]}) -> output/diag/plate_crop.jpg]")
    for sz in (256, 320, 384, 480, 640):
        os.environ["ANPR_CHAR_IMGSZ"] = str(sz)
        t_s = time.perf_counter()
        txt, conf_v, l1 = A.read_plate_with_char_model(crop_p, conf=0.15)
        ms = (time.perf_counter() - t_s) * 1000.0
        fused = A.refine_indonesian_plate(txt) if txt else ""
        print(f"  imgsz={sz:3d} -> {ms:5.0f}ms  raw={txt!r:10s} fused={fused!r:12s} conf={conf_v:.2f} n={len(l1)}", flush=True)
    os.environ["ANPR_CHAR_IMGSZ"] = "384"

    t_r0 = time.perf_counter()
    r_txt, r_cnf, r_all = A.rapidocr_read_plate(crop_p, return_all=True)
    r_ms = (time.perf_counter() - t_r0) * 1000.0
    print(f"  [RapidOCR ONNX] -> {r_ms:5.0f}ms  raw={r_txt!r} all={r_all} conf={r_cnf:.2f}", flush=True)

    t_e0 = time.perf_counter()
    ens_res = A.ensemble_plate_reading(crop_p)
    e_ms = (time.perf_counter() - t_e0) * 1000.0
    print(f"  [Hybrid Ensemble] -> {e_ms:5.0f}ms  final={ens_res.get('final')!r} method={ens_res.get('method')} conf={ens_res.get('confidence'):.2f}", flush=True)
else:
    sample_car_plate = os.path.join(ROOT, "test", "images", "New folder", "New folder", "IN_WSMPM1883119253389.png")
    if os.path.exists(sample_car_plate) and os.path.abspath(img_path) != os.path.abspath(sample_car_plate):
        print("\n" + "=" * 80)
        print("UJI OTOMATIS PADA FOTO MOBIL DENGAN PLAT JELAS (B 1542 DOL)")
        print("=" * 80)
        img_sample = cv2.imread(sample_car_plate)
        if img_sample is not None:
            A.confirmation_manager.tracks.clear()
            t_s0 = time.perf_counter()
            res_s = A.run_anpr(sample_car_plate, frame_img=img_sample.copy(), is_stream=False, frame_id=999, single_vehicle_mode=True)
            s_ms = (time.perf_counter() - t_s0) * 1000.0
            dets_s = res_s.get("detections") or []
            d_s = dets_s[0] if dets_s else {}
            print(f"[HASIL UJI B 1542 DOL] Total={s_ms:.1f}ms | vehicle={d_s.get('vehicle_type')} | body={d_s.get('body_style')} "
                  f"| plate_bbox={d_s.get('plate_bbox')} | license_plate={d_s.get('license_plate')!r} "
                  f"| method={d_s.get('ocr_method')} | conf={d_s.get('plate_confidence')}", flush=True)

print("\nSELESAI. Semua gambar diagnostik telah disimpan di folder output/diag/.", flush=True)
