"""
Simulator & Verifier otomatis kontrak Parkour <-> ALPR (Bagian C: S1-S10, Distribusi Durasi n>=30, dan Langkah 0).
Menjalankan HTTP server lokal di atas backend/app.py dengan ANPR_DEBUG_PARKOUR_LOG=1 tanpa mengubah perilaku produksi.
Semua plat disamarkan di output laporan dan log.
"""
import glob
import json
import os
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from werkzeug.serving import make_server

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

AUDIT_LOG_FILE = os.path.join(ROOT_DIR, "output", "parkour_audit.jsonl")
os.environ["ANPR_DEBUG_PARKOUR_LOG"] = "1"
os.environ["ANPR_PARKOUR_LOG_FILE"] = AUDIT_LOG_FILE
os.environ["ANPR_AUTH_DISABLED"] = "1"
os.environ["DISABLE_PADDLE"] = "1"
os.environ["SKIP_WARMUP"] = "1"

import cv2
import numpy as np
from tools.audit import parkour_audit_logger as pk_log
from backend import app as A


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return round(float(s[f]), 2)
    return round(float(s[f] + (s[c] - s[f]) * (k - f)), 2)


def stats_summary(values):
    if not values:
        return {"n": 0, "min_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "n": len(values),
        "min_ms": round(float(min(values)), 2),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": round(float(max(values)), 2),
    }


class ParkourHttpClient:
    """
    Meniru klien Parkour legacy (src/service.rs): /get_plate (timeout 2.0s), /get_image, /delete_image.
    Menjalankan setiap HTTP request di worker thread terpisah sehingga bila server > timeout_sec,
    klien mengalami TimeoutError tepat pada batas waktu sementara thread server masih lanjut berjalan (uji K6/S5).
    """
    def __init__(self, app_obj, timeout_sec=2.0):
        self.app_obj = app_obj
        self.timeout_sec = timeout_sec

    def call_json(self, method, path, payload=None, timeout_sec=None):
        t0 = time.perf_counter()
        to = timeout_sec if timeout_sec is not None else self.timeout_sec
        res_holder = {}
        done_evt = threading.Event()

        def _worker():
            with self.app_obj.test_client() as c:
                fn = getattr(c, method.lower())
                resp = fn(path, json=payload) if payload is not None else fn(path)
                raw = resp.get_data(as_text=False)
                res_holder["status"] = resp.status_code
                res_holder["raw"] = raw
                res_holder["json"] = resp.get_json(silent=True)
                done_evt.set()

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        finished = done_evt.wait(timeout=to)
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if not finished:
            return {
                "ok": False,
                "timeout": True,
                "status": None,
                "duration_ms": dur_ms,
                "error": f"Client timeout after {dur_ms:.1f} ms (limit {to*1000:.0f} ms)",
                "json": None,
            }
        return {
            "ok": True,
            "timeout": False,
            "status": res_holder["status"],
            "duration_ms": dur_ms,
            "json": res_holder["json"],
            "bytes": len(res_holder["raw"]),
        }

    def call_bytes(self, method, path, payload=None, timeout_sec=None):
        t0 = time.perf_counter()
        to = timeout_sec if timeout_sec is not None else self.timeout_sec
        with self.app_obj.test_client() as c:
            fn = getattr(c, method.lower())
            resp = fn(path, json=payload) if payload is not None else fn(path)
            raw = resp.get_data(as_text=False)
            dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "ok": resp.status_code == 200,
                "status": resp.status_code,
                "content_type": resp.mimetype,
                "duration_ms": dur_ms,
                "bytes": len(raw),
                "raw": raw,
            }


def extract_plate_and_file(resp_json):
    if not isinstance(resp_json, dict):
        return "", ""
    data = resp_json.get("data") or {}
    rec = None
    if isinstance(data.get("alpr"), list) and data["alpr"]:
        rec = data["alpr"][0]
    elif isinstance(data.get("capture"), dict):
        rec = data["capture"]
    if not isinstance(rec, dict):
        return "", ""
    plates = (rec.get("alpr") or {}).get("plates") or []
    plate_txt = plates[0].get("text", "") if plates else ""
    img_file = rec.get("image_file") or rec.get("alpr_file") or ""
    return plate_txt, img_file


def reset_engine_state(empty_frame, worker_fresh=True):
    with A.confirmation_manager.lock:
        A.confirmation_manager.tracks.clear()
    A.LATEST_RECORDS.clear()
    A.LAST_RECORDED_PLATES.clear()
    A.LAST_RECORDED_TRACK_IDS.clear()
    A.LAST_ALPR_RESPONSE_IMAGE["path"] = ""
    A.camera_stream_manager.running = True
    A.camera_stream_manager.latest_frame = empty_frame
    A.camera_stream_manager.latest_frame_time = time.time()
    A.camera_stream_manager.get_latest_frame = lambda: empty_frame
    A.stream_inference_worker.running = worker_fresh
    A.stream_inference_worker.last_payload = (
        {"timestamp": time.time(), "detections": []} if worker_fresh else None
    )


def inject_live_vehicle(tid, plate_display, vtype="car", body="MPV", conf=0.94, age_sec=0.05, state="ACTIVE", missing=0):
    with A.confirmation_manager.lock:
        trk = A.VehicleTrack(tid, {"bbox": [200, 150, 700, 520], "inside_interest_area": True})
        trk.lifecycle_state = state
        trk.frames_missing = missing
        trk.inside_interest_area = True
        trk.status = "CONFIRMED" if plate_display else "ANALYZING"
        trk.best_candidate_display = plate_display
        trk.best_candidate_conf = conf if plate_display else 0.0
        trk.cached_vtype = vtype
        trk.cached_body_style = body
        trk.last_seen = time.time() - age_sec
        trk.last_bbox = [200, 150, 700, 520]
        trk.last_plate_bbox = [380, 410, 510, 465] if plate_display else None
        A.confirmation_manager.tracks[tid] = trk
        return trk


def read_last_audit_entries(n=1):
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    http_lines = [x for x in lines if x.get("event") == "http_request"]
    return http_lines[-n:]


def run_all():
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as _f:
            pass

    commit_h, branch_n = pk_log.get_git_info()
    pk_log.emit_json({
        "event": "audit_session_start",
        "commit": commit_h,
        "branch": branch_n,
        "started_at": pk_log.now_iso_tz(),
    })

    client = ParkourHttpClient(A.app, timeout_sec=2.0)

    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    car_a_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    car_b_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    orig_run_anpr = A.run_anpr

    report = {
        "commit": commit_h,
        "branch": branch_n,
        "started_at": pk_log.now_iso_tz(),
        "step0": {},
        "scenarios": {},
        "latency_distributions": {},
    }

    try:
        # =========================================================================
        # LANGKAH 0: Uji Path Traversal & Deteksi Kendaraan / Kardus (0.48 vs 0.30)
        # =========================================================================
        traversal_Payloads = [
            "../../Windows/win.ini",
            "..\\..\\Windows\\win.ini",
            "../../etc/passwd",
            "C:\\Windows\\win.ini",
            "../backend/app.py",
            "latest_result.jpg",
            "..",
            ".",
        ]
        trav_get = []
        trav_del = []
        for pay in traversal_Payloads:
            r_g = client.call_bytes("POST", "/get_image", {"image_path": pay})
            r_d = client.call_json("POST", "/delete_image", {"image_paths": [pay]})
            last_e = read_last_audit_entries(2)
            trav_get.append({
                "payload": pay,
                "status": r_g["status"],
                "bytes": r_g["bytes"],
                "result_source": last_e[-2]["result_source"] if len(last_e) >= 2 else "",
                "request_id": last_e[-2]["request_id"] if len(last_e) >= 2 else "",
            })
            trav_del.append({
                "payload": pay,
                "status": r_d["status"],
                "delete_details": last_e[-1].get("delete_details") if last_e else [],
                "request_id": last_e[-1]["request_id"] if last_e else "",
            })
        report["step0"]["path_traversal_get_image"] = trav_get
        report["step0"]["path_traversal_delete_image"] = trav_del

        # Uji model kendaraan asli pada kardus, kertas foto Alphard, dan mobil/bus nyata
        artifact_dir = r"C:\Users\Lenovo\.gemini\antigravity\brain\cbb21041-1107-4e15-8424-1db246fed0e7\.user_uploaded"
        sample_files = {
            "kardus_room_1": os.path.join(artifact_dir, "media_1794904737869_7c06b911.jpg"),
            "kardus_room_2": os.path.join(artifact_dir, "media_1794907850972_9e9a3190.jpg"),
            "alphard_paper_chair": os.path.join(artifact_dir, "media_1794985341782_178d009b.jpg"),
            "cctv_gate_car": os.path.join(ROOT_DIR, "test", "images", "cctv_test_frame.jpg"),
            "real_bus": os.path.join(ROOT_DIR, "test", "images", "bus1.jpg"),
            "real_sedan": os.path.join(ROOT_DIR, "test", "images", "car sedan.jpg"),
        }
        veh_eval = {}
        for label, fpath in sample_files.items():
            if os.path.exists(fpath):
                im = cv2.imread(fpath)
                if im is not None:
                    with A.vehicle_infer_lock:
                        pred = A.vehicle_model.predict(im, conf=0.25, imgsz=384, device=A.DEVICE, verbose=False)[0]
                    boxes_info = []
                    for b in pred.boxes:
                        c_val = round(float(b.conf[0]), 4)
                        v_t = A.map_vehicle_class_name(int(b.cls[0]), A.vehicle_model)
                        boxes_info.append({
                            "type": v_t,
                            "conf": c_val,
                            "passes_048": c_val >= 0.48,
                            "passes_038_motor": c_val >= 0.38 if v_t == "motorcycle" else c_val >= 0.48,
                            "passes_old_030_dark": c_val >= 0.30,
                        })
                    veh_eval[label] = {"boxes": boxes_info}
        report["step0"]["vehicle_threshold_eval"] = veh_eval

        # =========================================================================
        # S1: Adegan kosong, 10 request (jarak simulasi 1, 5, 20, 60 dtk)
        # =========================================================================
        reset_engine_state(empty_frame, worker_fresh=True)
        s1_intervals = [1, 1, 5, 5, 20, 20, 60, 60, 1, 5]
        s1_results = []
        for idx, gap_s in enumerate(s1_intervals):
            # Simulasikan stream worker tetap hidup memperbarui payload saat kamera kosong
            A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": []}
            r = client.call_json("POST", "/get_plate", {})
            plate_txt, img_f = extract_plate_and_file(r["json"])
            ae = read_last_audit_entries(1)[0]
            s1_results.append({
                "seq": idx + 1,
                "simulated_gap_sec": gap_s,
                "duration_ms": r["duration_ms"],
                "plate": pk_log.mask_plate(plate_txt),
                "result_source": ae["result_source"],
                "request_id": ae["request_id"],
            })
        report["scenarios"]["S1"] = s1_results

        # =========================================================================
        # S2: A -> request -> hasil A. Lalu adegan kosong:
        #     (a) bila track A baru hilang 0.4s lalu dan masih berstatus ACTIVE (uji celah K4/K2 < 1.5s)
        #     (b) bila track A sudah EXITING / > 1.5s (< 5 dtk, mis. 2.0s) dan sesudah > 20 dtk
        # =========================================================================
        reset_engine_state(car_a_frame, worker_fresh=True)
        inject_live_vehicle(101, "B 1786 CIC", vtype="car", body="Sedan", conf=0.95, age_sec=0.05)
        r_s2_a = client.call_json("POST", "/get_plate", {})
        ae_s2_a = read_last_audit_entries(1)[0]

        # Sub-case S2a: 0.4s kemudian, kamera baru diarahkan ke kosong tetapi worker belum mengganti state ACTIVE (< 1.5s)
        with A.confirmation_manager.lock:
            A.confirmation_manager.tracks[101].last_seen = time.time() - 0.4
        r_s2_sub1 = client.call_json("POST", "/get_plate", {})
        ae_s2_sub1 = read_last_audit_entries(1)[0]

        # Sub-case S2b: 2.0s kemudian (< 5 dtk), kamera kosong (track > 1.5s / EXITING)
        with A.confirmation_manager.lock:
            A.confirmation_manager.tracks[101].last_seen = time.time() - 2.0
            A.confirmation_manager.tracks[101].lifecycle_state = "EXITING"
            A.confirmation_manager.tracks[101].frames_missing = 4
        A.camera_stream_manager.latest_frame = empty_frame
        A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": []}
        r_s2_sub2 = client.call_json("POST", "/get_plate", {})
        ae_s2_sub2 = read_last_audit_entries(1)[0]

        # Sub-case S2c: > 20 dtk kemudian (track expired)
        with A.confirmation_manager.lock:
            A.confirmation_manager.tracks.clear()
        A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": []}
        r_s2_sub3 = client.call_json("POST", "/get_plate", {})
        ae_s2_sub3 = read_last_audit_entries(1)[0]

        report["scenarios"]["S2"] = {
            "req_A": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s2_a["json"])[0]), "dur_ms": r_s2_a["duration_ms"], "req_id": ae_s2_a["request_id"]},
            "empty_after_0_4s_before_track_exit": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s2_sub1["json"])[0]), "dur_ms": r_s2_sub1["duration_ms"], "source": ae_s2_sub1["result_source"], "req_id": ae_s2_sub1["request_id"]},
            "empty_after_2_0s": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s2_sub2["json"])[0]), "dur_ms": r_s2_sub2["duration_ms"], "source": ae_s2_sub2["result_source"], "req_id": ae_s2_sub2["request_id"]},
            "empty_after_20s": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s2_sub3["json"])[0]), "dur_ms": r_s2_sub3["duration_ms"], "source": ae_s2_sub3["result_source"], "req_id": ae_s2_sub3["request_id"]},
        }

        # =========================================================================
        # S3: A -> request -> A. Lalu kendaraan B yang platnya tertutup/tidak terbaca -> harus kosong, bukan A (K3)
        # =========================================================================
        reset_engine_state(car_a_frame, worker_fresh=True)
        inject_live_vehicle(201, "B 1786 CIC", vtype="car", conf=0.95, age_sec=0.05)
        r_s3_a = client.call_json("POST", "/get_plate", {})
        ae_s3_a = read_last_audit_entries(1)[0]

        # Mobil A keluar, Mobil B masuk tapi plat tertutup (best_candidate_display = None)
        with A.confirmation_manager.lock:
            A.confirmation_manager.tracks[201].lifecycle_state = "EXITING"
            A.confirmation_manager.tracks[201].frames_missing = 5
        inject_live_vehicle(202, None, vtype="car", conf=0.0, age_sec=0.05)
        A.camera_stream_manager.latest_frame = car_b_frame
        A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": [{"track_id": 202, "status": "ANALYZING"}]}
        r_s3_b = client.call_json("POST", "/get_plate", {})
        ae_s3_b = read_last_audit_entries(1)[0]
        report["scenarios"]["S3"] = {
            "req_A": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s3_a["json"])[0]), "dur_ms": r_s3_a["duration_ms"], "req_id": ae_s3_a["request_id"]},
            "req_B_covered_plate": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s3_b["json"])[0]), "dur_ms": r_s3_b["duration_ms"], "source": ae_s3_b["result_source"], "req_id": ae_s3_b["request_id"]},
        }

        # =========================================================================
        # S4: A -> B (terdeteksi) -> hasil B. Request beruntun cepat & uji duplikasi
        # =========================================================================
        reset_engine_state(car_a_frame, worker_fresh=True)
        inject_live_vehicle(301, "B 1786 CIC", vtype="car", conf=0.95, age_sec=0.10)
        r_s4_a = client.call_json("POST", "/get_plate", {})
        ae_s4_a = read_last_audit_entries(1)[0]

        # B menggantikan A di kamera
        with A.confirmation_manager.lock:
            A.confirmation_manager.tracks[301].lifecycle_state = "EXITING"
            A.confirmation_manager.tracks[301].frames_missing = 4
        inject_live_vehicle(302, "B 2858 TII", vtype="car", conf=0.96, age_sec=0.02)
        r_s4_b1 = client.call_json("POST", "/get_plate", {})
        ae_s4_b1 = read_last_audit_entries(1)[0]
        # Tekan cepat ke-2 pada B (menguji apakah B dikembalikan dua kali karena consumed tidak dicek di _find_live_track)
        r_s4_b2 = client.call_json("POST", "/get_plate", {})
        ae_s4_b2 = read_last_audit_entries(1)[0]
        report["scenarios"]["S4"] = {
            "req_A": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s4_a["json"])[0]), "req_id": ae_s4_a["request_id"]},
            "req_B_first": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s4_b1["json"])[0]), "file": pk_log.mask_path_plates(os.path.basename(extract_plate_and_file(r_s4_b1["json"])[1])), "req_id": ae_s4_b1["request_id"]},
            "req_B_rapid_second": {"plate": pk_log.mask_plate(extract_plate_and_file(r_s4_b2["json"])[0]), "file": pk_log.mask_path_plates(os.path.basename(extract_plate_and_file(r_s4_b2["json"])[1])), "req_id": ae_s4_b2["request_id"]},
        }

        # =========================================================================
        # S5: Request A sengaja dibuat > 2.0 dtk (klien timeout 0.35s / 2.0s), lalu adegan kosong (K6)
        # =========================================================================
        reset_engine_state(car_a_frame, worker_fresh=False)
        def slow_run_anpr_a(frame, **kw):
            time.sleep(0.60)
            # Simulasikan juga efek OCR_RESCUE di background yang menghidupkan track saat selesai terlambat
            return {
                "detections": [{
                    "track_id": 401,
                    "license_plate": "B 1786 CIC",
                    "plate_confidence": 0.94,
                    "vehicle_type": "car",
                    "body_style": "Sedan",
                    "bbox": [200, 150, 700, 520],
                    "plate_bbox": [380, 410, 510, 465],
                    "is_newly_confirmed": True,
                    "status": "CONFIRMED",
                }],
                "detection_time_sec": 0.60,
            }
        A.run_anpr = slow_run_anpr_a
        # Klien menyerah lebih cepat (timeout=0.20s) sementara server selesai pada 0.60s
        r_s5_timeout = client.call_json("POST", "/get_plate", {}, timeout_sec=0.20)
        time.sleep(0.55)  # tunggu server menyelesaikan request terlambat A
        ae_s5_late = read_last_audit_entries(1)[0]
        leaked_last_img = os.path.basename(A.LAST_ALPR_RESPONSE_IMAGE.get("path") or "")

        # Sekarang kamera kosong, kirim request berikutnya
        A.camera_stream_manager.latest_frame = empty_frame
        A.stream_inference_worker.running = True
        A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": []}
        A.run_anpr = lambda frame, **kw: {"detections": [], "detection_time_sec": 0.05}
        r_s5_next = client.call_json("POST", "/get_plate", {})
        ae_s5_next = read_last_audit_entries(1)[0]
        report["scenarios"]["S5"] = {
            "req_1_client_timed_out": r_s5_timeout["timeout"],
            "req_1_server_finished_ms": ae_s5_late["duration_ms"],
            "req_1_server_req_id": ae_s5_late["request_id"],
            "late_result_saved_to_last_response_image": pk_log.mask_path_plates(leaked_last_img),
            "req_2_on_empty_plate": pk_log.mask_plate(extract_plate_and_file(r_s5_next["json"])[0]),
            "req_2_req_id": ae_s5_next["request_id"],
        }

        # =========================================================================
        # S6: Setelah A dikonsumsi -> /delete_image (path backslash), lalu request berikutnya (K4)
        # =========================================================================
        reset_engine_state(car_a_frame, worker_fresh=True)
        inject_live_vehicle(601, "B 1786 CIC", vtype="car", conf=0.95, age_sec=0.05)
        r_s6_a = client.call_json("POST", "/get_plate", {})
        _, s6_img_path = extract_plate_and_file(r_s6_a["json"])
        win_backslash_path = s6_img_path.replace("/", "\\")
        r_s6_del = client.call_json("POST", "/delete_image", {"image_paths": [win_backslash_path]})
        ae_s6_del = read_last_audit_entries(1)[0]

        # Cek 1: Jika mobil A masih terlihat (< 1.5s) sesudah /delete_image, apakah masih dikembalikan?
        r_s6_still_there = client.call_json("POST", "/get_plate", {})
        ae_s6_still = read_last_audit_entries(1)[0]

        # Cek 2: Jika adegan sudah kosong sesudah /delete_image
        with A.confirmation_manager.lock:
            A.confirmation_manager.tracks.clear()
        A.camera_stream_manager.latest_frame = empty_frame
        A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": []}
        r_s6_empty = client.call_json("POST", "/get_plate", {})
        ae_s6_empty = read_last_audit_entries(1)[0]

        report["scenarios"]["S6"] = {
            "delete_backslash_status": r_s6_del["status"],
            "delete_req_id": ae_s6_del["request_id"],
            "file_deleted": not os.path.exists(s6_img_path),
            "next_req_if_track_still_within_1_5s": {
                "plate": pk_log.mask_plate(extract_plate_and_file(r_s6_still_there["json"])[0]),
                "req_id": ae_s6_still["request_id"],
            },
            "next_req_on_empty_scene": {
                "plate": pk_log.mask_plate(extract_plate_and_file(r_s6_empty["json"])[0]),
                "req_id": ae_s6_empty["request_id"],
            },
        }

        # =========================================================================
        # S7: /delete_image tanpa get_image, file tidak ada, "..", path absolut, latest_result.jpg
        # =========================================================================
        latest_res_file = os.path.join(A.CAPTURES_DIR, "latest_result.jpg")
        cv2.imwrite(latest_res_file, empty_frame)
        s7_payloads = [
            "non_existent_file_12345.jpg",
            "../../Windows/win.ini",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "latest_result.jpg",
            "..",
        ]
        r_s7_1 = client.call_json("DELETE", "/delete_image", {"image_paths": s7_payloads})
        ae_s7_1 = read_last_audit_entries(1)[0]
        r_s7_2 = client.call_json("POST", "/delete_image", {"image_paths": s7_payloads})
        ae_s7_2 = read_last_audit_entries(1)[0]
        report["scenarios"]["S7"] = {
            "call_1_status": r_s7_1["status"],
            "call_1_req_id": ae_s7_1["request_id"],
            "call_2_idempotent_status": r_s7_2["status"],
            "call_2_req_id": ae_s7_2["request_id"],
            "latest_result_still_exists": os.path.exists(latest_res_file),
            "details": ae_s7_1.get("delete_details"),
        }

        # =========================================================================
        # S8: /get_image untuk file yang sudah dihapus -> dummy 200 JPEG
        # =========================================================================
        r_s8 = client.call_bytes("POST", "/get_image", {"image_path": win_backslash_path})
        ae_s8 = read_last_audit_entries(1)[0]
        decoded_s8 = cv2.imdecode(np.frombuffer(r_s8["raw"], dtype=np.uint8), cv2.IMREAD_COLOR) if r_s8.get("raw") else None
        report["scenarios"]["S8"] = {
            "status_code": r_s8["status"],
            "content_type": r_s8["content_type"],
            "bytes": r_s8["bytes"],
            "dimensions": list(decoded_s8.shape) if decoded_s8 is not None else None,
            "mean_pixel_intensity": round(float(decoded_s8.mean()), 2) if decoded_s8 is not None else None,
            "result_source": ae_s8["result_source"],
            "req_id": ae_s8["request_id"],
        }

        # =========================================================================
        # S9: Mode "engine baru nyala (warm)" vs "request pertama setelah idle 5 menit"
        # =========================================================================
        A.run_anpr = orig_run_anpr
        reset_engine_state(empty_frame, worker_fresh=True)
        r_s9_warm = client.call_json("POST", "/get_plate", {})
        ae_s9_warm = read_last_audit_entries(1)[0]

        # Simulasikan last_payload berusia > 5 menit (305 detik) sehingga jatuh ke cold run_anpr
        A.stream_inference_worker.last_payload = {"timestamp": time.time() - 305.0, "detections": []}
        r_s9_idle = client.call_json("POST", "/get_plate", {})
        ae_s9_idle = read_last_audit_entries(1)[0]
        report["scenarios"]["S9"] = {
            "warm_empty_ms": r_s9_warm["duration_ms"],
            "warm_source": ae_s9_warm["result_source"],
            "warm_req_id": ae_s9_warm["request_id"],
            "after_5min_idle_ms": r_s9_idle["duration_ms"],
            "after_5min_idle_source": ae_s9_idle["result_source"],
            "after_5min_idle_req_id": ae_s9_idle["request_id"],
        }

        # =========================================================================
        # S10: Kardus / objek bukan kendaraan dengan ambang terbaru (0.48)
        # =========================================================================
        kardus_path = sample_files["kardus_room_2"]
        kardus_img = cv2.imread(kardus_path) if os.path.exists(kardus_path) else empty_frame
        reset_engine_state(kardus_img, worker_fresh=False)
        r_s10 = client.call_json("POST", "/get_plate", {})
        ae_s10 = read_last_audit_entries(1)[0]
        s10_plate, _ = extract_plate_and_file(r_s10["json"])
        report["scenarios"]["S10"] = {
            "plate": pk_log.mask_plate(s10_plate),
            "duration_ms": r_s10["duration_ms"],
            "result_source": ae_s10["result_source"],
            "req_id": ae_s10["request_id"],
        }

        # =========================================================================
        # DISTRIBUSI DURASI (n = 35 >= 30 per kondisi)
        # =========================================================================
        # 1a. Ada kendaraan (live stream track aktif)
        d_veh_live = []
        reset_engine_state(car_a_frame, worker_fresh=True)
        for i in range(35):
            inject_live_vehicle(700 + i, "B 1786 CIC", vtype="car", conf=0.95, age_sec=0.03)
            r = client.call_json("POST", "/get_plate", {})
            d_veh_live.append(r["duration_ms"])
            with A.confirmation_manager.lock:
                A.confirmation_manager.tracks.clear()

        # 1b. Ada kendaraan (cold scan run_anpr nyata pada gambar kendaraan)
        real_car_path = sample_files["cctv_gate_car"]
        real_car_img = cv2.imread(real_car_path) if os.path.exists(real_car_path) else car_a_frame
        d_veh_cold = []
        reset_engine_state(real_car_img, worker_fresh=False)
        for _ in range(35):
            with A.confirmation_manager.lock:
                A.confirmation_manager.tracks.clear()
            r = client.call_json("POST", "/get_plate", {})
            d_veh_cold.append(r["duration_ms"])

        # 2a. Tidak ada kendaraan (stream worker aktif)
        d_empty_live = []
        reset_engine_state(empty_frame, worker_fresh=True)
        for _ in range(35):
            A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": []}
            r = client.call_json("POST", "/get_plate", {})
            d_empty_live.append(r["duration_ms"])

        # 2b. Tidak ada kendaraan (cold scan run_anpr nyata pada frame kosong)
        d_empty_cold = []
        reset_engine_state(empty_frame, worker_fresh=False)
        for _ in range(35):
            r = client.call_json("POST", "/get_plate", {})
            d_empty_cold.append(r["duration_ms"])

        # 3a. Ada kendaraan tetapi plat tidak terbaca / tertutup (stream pending wait 0.8s)
        d_unreadable_wait = []
        reset_engine_state(car_b_frame, worker_fresh=True)
        for i in range(35):
            inject_live_vehicle(900 + i, None, vtype="car", conf=0.0, age_sec=0.02)
            A.stream_inference_worker.last_payload = {"timestamp": time.time(), "detections": [{"track_id": 900 + i}]}
            r = client.call_json("POST", "/get_plate", {})
            d_unreadable_wait.append(r["duration_ms"])
            with A.confirmation_manager.lock:
                A.confirmation_manager.tracks.clear()

        report["latency_distributions"] = {
            "ada_kendaraan_live_track": stats_summary(d_veh_live),
            "ada_kendaraan_cold_scan": stats_summary(d_veh_cold),
            "tidak_ada_kendaraan_stream_aktif": stats_summary(d_empty_live),
            "tidak_ada_kendaraan_cold_scan": stats_summary(d_empty_cold),
            "plat_tidak_terbaca_pending_wait": stats_summary(d_unreadable_wait),
        }

    finally:
        A.run_anpr = orig_run_anpr
        # Bersihkan file snapshot sintetis hasil pengujian dari CAPTURES_DIR
        for f in glob.glob(os.path.join(A.CAPTURES_DIR, "ALPR_2026*")):
            try:
                os.remove(f)
            except Exception:
                pass

    out_json = os.path.join(ROOT_DIR, "output", "parkour_simulation_report.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_all()
