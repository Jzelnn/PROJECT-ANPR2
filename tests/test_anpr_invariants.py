#!/usr/bin/env python3
"""
Automated Invariant & Regression Tests for PROJECT-ANPR2
========================================================
Menguji:
1. Aturan validasi & pembersihan plat nomor Indonesia (refine_indonesian_plate & is_valid_indonesian_plate_structure).
2. Track ID unik pada pemrosesan foto beruntun (tanpa bentrok ID).
3. Anti-overwrite pada penyimpanan riwayat (LATEST_RECORDS / parking_history.json).
4. Konsistensi struktur data antara keluaran run_anpr, kartu UI, dan catatan riwayat.
5. Ketahanan terhadap race condition saat pembaruan bodi asinkron (_async_body_task).
"""

import os
import sys
import time
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("SKIP_WARMUP", "1")
os.environ.setdefault("DISABLE_PADDLE", "1")

import app as A


class TestAnprInvariants(unittest.TestCase):

    def test_01_indonesian_plate_validation_and_refine(self):
        """Pastikan refine_indonesian_plate dan is_valid_indonesian_plate_structure tetap utuh."""
        self.assertEqual(A.refine_indonesian_plate("B1542DOL"), "B 1542 DOL")
        self.assertEqual(A.refine_indonesian_plate("84745SRT"), "B 4745 SRT")
        self.assertTrue(A.is_valid_indonesian_plate_structure("B 1542 DOL")[0])
        self.assertTrue(A.is_valid_indonesian_plate_structure("523-07")[0])
        self.assertFalse(A.is_valid_indonesian_plate_structure("")[0])
        self.assertFalse(A.is_valid_indonesian_plate_structure("07 28")[0])

    def test_02_unique_track_ids_on_consecutive_photos(self):
        """Pastikan foto beruntun selalu mendapatkan Track ID unik yang monoton naik."""
        seen_tids = []
        for _ in range(5):
            det = {
                "track_id": None,
                "license_plate": "B 1542 DOL",
                "plate_confidence": 0.92,
                "vehicle_type": "car",
                "vehicle_confidence": 0.90,
                "bbox": [100, 100, 500, 400],
                "plate_bbox": [220, 310, 340, 355],
            }
            out = A.confirmation_manager.update([det], is_stream=False)
            self.assertEqual(len(out), 1)
            tid = out[0].get("track_id")
            self.assertIsNotNone(tid)
            self.assertNotIn(tid, seen_tids)
            seen_tids.append(tid)
        self.assertEqual(seen_tids, sorted(seen_tids))

    def test_03_anti_overwrite_history_and_sync_fields(self):
        """Pastikan dua kendaraan berbeda pada foto beruntun tidak saling menimpa di LATEST_RECORDS."""
        initial_len = len(A.LATEST_RECORDS)
        plates = ["B 1101 AAA", "D 2202 BBB", "F 3303 CCC"]
        saved_ids = []
        # Bersihkan cache anti-passback khusus untuk plat uji sintetis ini
        for p in plates:
            A.LAST_RECORDED_PLATES.pop(p.replace(" ", ""), None)

        for idx, p in enumerate(plates):
            tid = 90000 + idx
            A.LAST_RECORDED_TRACK_IDS.pop(tid, None)
            det = {
                "track_id": tid,
                "license_plate": p,
                "plate_confidence": 0.91,
                "ocr_confidence": 0.91,
                "vehicle_type": "car",
                "vehicle_confidence": 0.88,
                "body_style": "MPV",
                "body_style_confidence": 0.85,
                "status": "CONFIRMED",
                "is_newly_confirmed": True,
                "latency_ms": 420,
                "processing_ms": 420,
                "timing": {
                    "vehicle_ms": 180.0,
                    "plate_ms": 110.0,
                    "ocr_ms": 60.0,
                    "body_ms": 70.0,
                    "total_ms": 420.0,
                },
            }
            rec = A.save_parking_record(det, source_img=None)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["license_plate"], p)
            self.assertEqual(rec["track_id"], tid)
            self.assertEqual(rec["processing_ms"], 420)
            saved_ids.append(rec["id"])
            time.sleep(0.002)

        self.assertEqual(len(set(saved_ids)), 3)
        recent_plates = [r["license_plate"] for r in list(A.LATEST_RECORDS)[:3]]
        self.assertEqual(recent_plates, list(reversed(plates)))

        # Bersihkan kembali record sintetis dari LATEST_RECORDS agar tidak mengotori state
        while len(A.LATEST_RECORDS) > initial_len:
            A.LATEST_RECORDS.popleft()

    def test_04_async_body_race_on_consecutive_tracks(self):
        """Pastikan penyelesaian bodi asinkron mengisi baris dengan Track ID yang tepat tanpa tertukar."""
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        t1, t2 = 88001, 88002
        rec1 = {
            "id": 77001,
            "track_id": t1,
            "timestamp": "2026-11-11 10:00:01",
            "license_plate": "B 7001 RCE",
            "vehicle_type": "car",
            "body_style": None,
            "body_pending": True,
        }
        rec2 = {
            "id": 77002,
            "track_id": t2,
            "timestamp": "2026-11-11 10:00:02",
            "license_plate": "B 7002 RCE",
            "vehicle_type": "truck",
            "body_style": None,
            "body_pending": True,
        }
        A.LATEST_RECORDS.appendleft(rec1)
        A.LATEST_RECORDS.appendleft(rec2)
        try:
            # Jalankan _async_body_task secara terbalik (t2 selesai lebih dulu daripada t1)
            A._async_body_task(t2, dummy_img, [10, 10, 200, 200], "truck", 0.90, False, [])
            A._async_body_task(t1, dummy_img, [10, 10, 200, 200], "car", 0.90, False, [])
            self.assertFalse(rec1["body_pending"])
            self.assertFalse(rec2["body_pending"])
            self.assertIsNotNone(rec1["body_style"])
            self.assertIsNotNone(rec2["body_style"])
        finally:
            if rec2 in A.LATEST_RECORDS:
                A.LATEST_RECORDS.remove(rec2)
            if rec1 in A.LATEST_RECORDS:
                A.LATEST_RECORDS.remove(rec1)

    def test_05_security_token_cors_and_url_redaction(self):
        """Pastikan autentikasi token API, pembatasan CORS, validasi bind host, dan penyensoran password RTSP bekerja."""
        # 1. Redaksi URL RTSP tidak boleh membocorkan password
        raw_url = "rtsp://admin:SuperSecret123!@192.168.110.64:554/Streaming/Channels/101"
        redacted = A._redact_url(raw_url)
        self.assertEqual(redacted, "rtsp://admin:***@192.168.110.64:554/Streaming/Channels/101")
        self.assertNotIn("SuperSecret123!", redacted)

        # 2. Validasi startup bind host non-localhost
        old_tok = os.environ.get("ANPR_API_TOKEN")
        old_dis = os.environ.get("ANPR_AUTH_DISABLED")
        old_cors = os.environ.get("ANPR_CORS_ORIGINS")
        try:
            os.environ["ANPR_API_TOKEN"] = ""
            os.environ["ANPR_AUTH_DISABLED"] = "0"
            ok_0, _ = A.validate_startup_security_config("0.0.0.0")
            self.assertFalse(ok_0)
            ok_local, _ = A.validate_startup_security_config("127.0.0.1")
            self.assertTrue(ok_local)

            os.environ["ANPR_AUTH_DISABLED"] = "1"
            ok_dis, _ = A.validate_startup_security_config("0.0.0.0")
            self.assertTrue(ok_dis)

            # 3. Endpoint /api/* wajib menolak akses (401) saat ANPR_API_TOKEN disetel dan header kosong/salah
            os.environ["ANPR_AUTH_DISABLED"] = "0"
            os.environ["ANPR_API_TOKEN"] = "token-uji-rahasia-999"
            os.environ["ANPR_CORS_ORIGINS"] = "http://dashboard.rs.local:5001"
            client = A.app.test_client()

            r_unauth = client.get("/api/health")
            self.assertEqual(r_unauth.status_code, 401)

            r_clear_unauth = client.post("/api/clear_history")
            self.assertEqual(r_clear_unauth.status_code, 401)

            r_auth_bearer = client.get(
                "/api/health",
                headers={
                    "Authorization": "Bearer token-uji-rahasia-999",
                    "Origin": "http://dashboard.rs.local:5001",
                },
            )
            self.assertEqual(r_auth_bearer.status_code, 200)
            self.assertEqual(
                r_auth_bearer.headers.get("Access-Control-Allow-Origin"),
                "http://dashboard.rs.local:5001",
            )

            r_evil_origin = client.get(
                "/api/health",
                headers={
                    "X-API-Token": "token-uji-rahasia-999",
                    "Origin": "http://attacker.evil.com",
                },
            )
            self.assertEqual(r_evil_origin.status_code, 200)
            self.assertIsNone(r_evil_origin.headers.get("Access-Control-Allow-Origin"))
        finally:
            if old_tok is None:
                os.environ.pop("ANPR_API_TOKEN", None)
            else:
                os.environ["ANPR_API_TOKEN"] = old_tok
            if old_dis is None:
                os.environ.pop("ANPR_AUTH_DISABLED", None)
            else:
                os.environ["ANPR_AUTH_DISABLED"] = old_dis
            if old_cors is None:
                os.environ.pop("ANPR_CORS_ORIGINS", None)
            else:
                os.environ["ANPR_CORS_ORIGINS"] = old_cors

    def test_06_session_cookie_csrf_and_ws_origin(self):
        """
        Uji syarat Opsi A (Session Cookie + Proteksi CSRF + Validasi Origin WebSocket):
        1. Tanpa cookie / header -> 401
        2. Cookie kedaluwarsa -> 401
        3. Tanda tangan cookie dimanipulasi -> 401
        4. POST dengan cookie sah tetapi tanpa header Authorization/X-API-Token -> 401 (cegah CSRF)
        5. WebSocket dengan Origin asing -> ditolak
        6. GET /captures/* dengan cookie sesi sah -> 200
        """
        old_tok = os.environ.get("ANPR_API_TOKEN")
        old_dis = os.environ.get("ANPR_AUTH_DISABLED")
        old_cors = os.environ.get("ANPR_CORS_ORIGINS")
        dummy_cap_name = "_unit_test_dummy_capture.jpg"
        dummy_cap_path = os.path.join(A.CAPTURES_DIR, dummy_cap_name)

        try:
            os.environ["ANPR_AUTH_DISABLED"] = "0"
            os.environ["ANPR_API_TOKEN"] = "token-uji-opsi-a-777"
            os.environ["ANPR_CORS_ORIGINS"] = "http://localhost:5001,http://127.0.0.1:5001"
            with A._AUTH_RATE_LOCK:
                A._AUTH_FAILED_ATTEMPTS.clear()

            # Siapkan file gambar dummy di CAPTURES_DIR untuk menguji GET /captures/<filename>
            dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
            A.cv2.imwrite(dummy_cap_path, dummy_img)

            client = A.app.test_client()

            # (1) Tanpa cookie dan tanpa header -> 401 pada /captures/* dan /api/export
            r_no_cookie = client.get(f"/captures/{dummy_cap_name}")
            self.assertEqual(r_no_cookie.status_code, 401)
            r_export_no_cookie = client.get("/api/export")
            self.assertEqual(r_export_no_cookie.status_code, 401)

            # (2) Cookie kedaluwarsa -> 401
            expired_cookie = A.create_session_cookie_value(ttl_sec=-10)
            self.assertFalse(A.verify_session_cookie_value(expired_cookie))
            client.set_cookie(A.SESSION_COOKIE_NAME, expired_cookie)
            r_expired = client.get(f"/captures/{dummy_cap_name}")
            self.assertEqual(r_expired.status_code, 401)

            # (3) Tanda tangan cookie dimanipulasi -> 401
            valid_cookie = A.create_session_cookie_value(ttl_sec=3600)
            self.assertTrue(A.verify_session_cookie_value(valid_cookie))
            tampered_cookie = valid_cookie[:-4] + ("0000" if not valid_cookie.endswith("0000") else "ffff")
            self.assertFalse(A.verify_session_cookie_value(tampered_cookie))
            client.set_cookie(A.SESSION_COOKIE_NAME, tampered_cookie)
            r_tampered = client.get(f"/captures/{dummy_cap_name}")
            self.assertEqual(r_tampered.status_code, 401)

            # Tukarkan token melalui POST /api/auth/session untuk mendapatkan cookie sesi resmi
            r_login = client.post(
                "/api/auth/session",
                headers={"X-API-Token": "token-uji-opsi-a-777"},
            )
            self.assertEqual(r_login.status_code, 200)
            set_cookie_hdr = r_login.headers.get("Set-Cookie", "")
            self.assertIn("HttpOnly", set_cookie_hdr)
            self.assertIn("SameSite=Lax", set_cookie_hdr)

            # (6) GET /captures/<file> dengan cookie sesi sah -> 200
            r_cap_ok = client.get(f"/captures/{dummy_cap_name}")
            self.assertEqual(r_cap_ok.status_code, 200)

            # (4) POST dengan cookie sesi sah tetapi TANPA header Authorization/X-API-Token -> 401 (proteksi CSRF)
            r_csrf_post = client.post("/api/clear_history")
            self.assertEqual(r_csrf_post.status_code, 401)
            r_csrf_stream = client.post("/api/stream/stop")
            self.assertEqual(r_csrf_stream.status_code, 401)

            # (5) WebSocket dengan Origin asing wajib ditolak
            with A.app.test_request_context(
                "/ws/live",
                headers={"Origin": "http://evil-foreign-site.example.com"},
            ):
                self.assertFalse(A._is_origin_allowed(A.request))

                class _DummyWS:
                    def __init__(self):
                        self.sent = []
                        self.closed = False
                    def send(self, data):
                        self.sent.append(data)
                    def close(self):
                        self.closed = True

                dws = _DummyWS()
                A.live_ws(dws)
                self.assertTrue(dws.closed)
                self.assertTrue(any("Forbidden WebSocket Origin" in m for m in dws.sent))

            # Uji Logout menghapus cookie sesi
            r_logout = client.post("/api/auth/logout")
            self.assertEqual(r_logout.status_code, 200)
            r_after_logout = client.get(f"/captures/{dummy_cap_name}")
            self.assertEqual(r_after_logout.status_code, 401)

        finally:
            if os.path.exists(dummy_cap_path):
                try:
                    os.remove(dummy_cap_path)
                except OSError:
                    pass
            if old_tok is None:
                os.environ.pop("ANPR_API_TOKEN", None)
            else:
                os.environ["ANPR_API_TOKEN"] = old_tok
            if old_dis is None:
                os.environ.pop("ANPR_AUTH_DISABLED", None)
            else:
                os.environ["ANPR_AUTH_DISABLED"] = old_dis
            if old_cors is None:
                os.environ.pop("ANPR_CORS_ORIGINS", None)
            else:
                os.environ["ANPR_CORS_ORIGINS"] = old_cors

    def test_07_pure_arm_platform_detection(self):
        """
        Pastikan logika should_enable_pure_arm sesuai spesifikasi Langkah 2:
        - Aktif di Linux aarch64/arm64 saat ANPR_PURE_ARM kosong atau "1"
        - Mati di Linux aarch64/arm64 saat ANPR_PURE_ARM="0"
        - Mati di macOS (darwin arm64) dan Windows/Linux x86_64 secara default
        - Aktif di mana saja jika ANPR_PURE_ARM="1"
        """
        self.assertTrue(A.should_enable_pure_arm("linux", "aarch64", ""))
        self.assertTrue(A.should_enable_pure_arm("linux", "arm64", ""))
        self.assertFalse(A.should_enable_pure_arm("linux", "aarch64", "0"))
        self.assertFalse(A.should_enable_pure_arm("darwin", "arm64", ""))
        self.assertFalse(A.should_enable_pure_arm("win32", "AMD64", ""))
        self.assertFalse(A.should_enable_pure_arm("linux", "x86_64", ""))
        self.assertTrue(A.should_enable_pure_arm("win32", "AMD64", "1"))

    def test_08_pure_arm_and_official_predict_equivalence(self):
        """
        Uji kesetaraan numerik & struktural pure_arm_yolo_predict vs predict resmi Ultralytics:
        - Jumlah deteksi sama
        - IoU bounding box > 0.95
        - Selisih confidence < 0.02
        Diuji pada crop kendaraan, crop plat nomor, dan model karakter.
        """
        def _calc_iou(b1, b2):
            xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
            xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
            a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            return inter / float(a1 + a2 - inter)

        # Uji pada beberapa foto uji nyata di test/images
        test_imgs = [
            os.path.join(ROOT, "test", "images", "New folder", "New folder", "IN_WSMPM1883119253389.png"),
            os.path.join(ROOT, "test", "images", "New folder", "New folder", "OUT_WSMPM1883118474851.png"),
            os.path.join(ROOT, "test", "images", "WhatsApp Image 2026-09-04 at 16.47.50.jpeg"),
        ]
        tested_any = False
        for img_path in test_imgs:
            if not os.path.isfile(img_path):
                continue
            img = A.cv2.imread(img_path)
            if img is None:
                continue
            tested_any = True

            # 1. Evaluasi Vehicle Model
            A.set_pure_arm_mode(False)
            v_off = A.vehicle_model(img, imgsz=384, conf=0.35, verbose=False)
            A.set_pure_arm_mode(True)
            v_arm = A.vehicle_model(img, imgsz=384, conf=0.35, verbose=False)

            self.assertEqual(len(v_off[0].boxes), len(v_arm[0].boxes))
            for b_o, b_a in zip(v_off[0].boxes, v_arm[0].boxes):
                self.assertEqual(int(b_o.cls[0]), int(b_a.cls[0]))
                self.assertLessEqual(abs(float(b_o.conf[0]) - float(b_a.conf[0])), 0.02)
                iou = _calc_iou([float(x) for x in b_o.xyxy[0]], [float(x) for x in b_a.xyxy[0]])
                self.assertGreaterEqual(iou, 0.95)

            # 2. Evaluasi Plate Model pada crop kendaraan
            if len(v_off[0].boxes) > 0:
                vbox = [int(v) for v in v_off[0].boxes[0].xyxy[0]]
                v_crop = img[vbox[1]:vbox[3], vbox[0]:vbox[2]]
            else:
                v_crop = img

            A.set_pure_arm_mode(False)
            p_off = A.plate_detector(v_crop, imgsz=320, conf=0.12, verbose=False)
            A.set_pure_arm_mode(True)
            p_arm = A.plate_detector(v_crop, imgsz=320, conf=0.12, verbose=False)

            self.assertEqual(len(p_off[0].boxes), len(p_arm[0].boxes))
            for b_o, b_a in zip(p_off[0].boxes, p_arm[0].boxes):
                self.assertEqual(int(b_o.cls[0]), int(b_a.cls[0]))
                self.assertLessEqual(abs(float(b_o.conf[0]) - float(b_a.conf[0])), 0.02)
                iou = _calc_iou([float(x) for x in b_o.xyxy[0]], [float(x) for x in b_a.xyxy[0]])
                self.assertGreaterEqual(iou, 0.95)

            # 3. Evaluasi Char Model pada crop plat nomor
            if len(p_off[0].boxes) > 0:
                pbox = [int(v) for v in p_off[0].boxes[0].xyxy[0]]
                p_crop = v_crop[pbox[1]:pbox[3], pbox[0]:pbox[2]]
            else:
                p_crop = v_crop

            A.set_pure_arm_mode(False)
            c_off = A.char_model(p_crop, imgsz=384, conf=0.10, iou=0.45, verbose=False)
            A.set_pure_arm_mode(True)
            c_arm = A.char_model(p_crop, imgsz=384, conf=0.10, iou=0.45, verbose=False)

            self.assertEqual(len(c_off[0].boxes), len(c_arm[0].boxes))
            for b_o, b_a in zip(c_off[0].boxes, c_arm[0].boxes):
                self.assertEqual(int(b_o.cls[0]), int(b_a.cls[0]))
                self.assertLessEqual(abs(float(b_o.conf[0]) - float(b_a.conf[0])), 0.02)
                iou = _calc_iou([float(x) for x in b_o.xyxy[0]], [float(x) for x in b_a.xyxy[0]])
                self.assertGreaterEqual(iou, 0.95)

        # Kembalikan mode pure_arm ke deteksi platform
        A.set_pure_arm_mode(A.should_enable_pure_arm())
        self.assertTrue(tested_any)

    def test_09_unreadable_blocks_history_and_can_be_disabled(self):
        """Plat tidak terkonfirmasi menjadi UNREADABLE dan tidak masuk history. ANPR_UNREADABLE=0 mengembalikan CONFIRMED."""
        old = os.environ.get("ANPR_UNREADABLE")
        try:
            os.environ["ANPR_UNREADABLE"] = "1"
            junk = {
                "track_id": 91001,
                "license_plate": "A",
                "plate_confidence": 0.95,
                "ocr_confidence": 0.95,
                "vehicle_type": "car",
                "vehicle_confidence": 0.90,
                "bbox": [10, 10, 200, 160],
            }
            out = A.confirmation_manager.update([junk], is_stream=False)
            self.assertEqual(out[0]["status"], "UNREADABLE")
            self.assertEqual(out[0]["ocr_status"], "UNREADABLE")
            self.assertFalse(out[0]["is_newly_confirmed"])
            self.assertIsNone(A.save_parking_record(out[0]))

            good = {
                "track_id": 91002,
                "license_plate": "B 1542 DOL",
                "plate_confidence": 0.91,
                "ocr_confidence": 0.91,
                "vehicle_type": "car",
                "vehicle_confidence": 0.90,
                "bbox": [10, 10, 200, 160],
            }
            A.LAST_RECORDED_PLATES.pop("B1542DOL", None)
            A.LAST_RECORDED_TRACK_IDS.pop(91002, None)
            out_ok = A.confirmation_manager.update([good], is_stream=False)
            self.assertEqual(out_ok[0]["status"], "CONFIRMED")
            self.assertTrue(out_ok[0]["is_newly_confirmed"])

            os.environ["ANPR_UNREADABLE"] = "0"
            legacy = dict(junk)
            legacy["track_id"] = 91003
            out_off = A.confirmation_manager.update([legacy], is_stream=False)
            self.assertEqual(out_off[0]["status"], "CONFIRMED")
            self.assertTrue(out_off[0]["is_newly_confirmed"])
            self.assertIsNone(A.save_parking_record(out_off[0]))
        finally:
            if old is None:
                os.environ.pop("ANPR_UNREADABLE", None)
            else:
                os.environ["ANPR_UNREADABLE"] = old

    def test_10_latency_schema_and_body_imgsz(self):
        """Key vehicle_ms tetap ada. ANPR_BODY_IMGSZ hanya mengubah ukuran, default 192."""
        perf = A.build_perf_breakdown(10, 20, 1, 0, 30, 61, 16.4, 9, 1)
        for key in ("vehicle_ms", "yolo_ms", "plate_ms", "ocr_ms", "body_ms", "total_ms"):
            self.assertIn(key, perf)
        self.assertEqual(perf["vehicle_ms"], perf["yolo_ms"])
        self.assertEqual(perf["body_ms"], 0.0)

        old = os.environ.get("ANPR_BODY_IMGSZ")
        try:
            os.environ.pop("ANPR_BODY_IMGSZ", None)
            self.assertEqual(A.body_infer_imgsz(), 192)
            os.environ["ANPR_BODY_IMGSZ"] = "224"
            self.assertEqual(A.body_infer_imgsz(), 224)
            self.assertTrue(A.body_async_on_photo())
            os.environ["ANPR_BODY_ASYNC"] = "0"
            self.assertFalse(A.body_async_on_photo())
        finally:
            os.environ.pop("ANPR_BODY_ASYNC", None)
            if old is None:
                os.environ.pop("ANPR_BODY_IMGSZ", None)
            else:
                os.environ["ANPR_BODY_IMGSZ"] = old

    def test_11_plate_row_crop_and_longer_string_does_not_win(self):
        """Baris pajak terpotong. String yang lebih panjang tidak menggantikan bacaan yang lebih yakin."""
        img = np.zeros((40, 120, 3), dtype=np.uint8)
        os.environ["ANPR_CHAR_ROW_RATIO"] = "0.74"
        row = A.plate_number_row(img, vehicle_type="car")
        self.assertEqual(row.shape[0], int(round(40 * 0.74)))
        os.environ["ANPR_CHAR_ROW_RATIO"] = "1"
        self.assertEqual(A.plate_number_row(img, vehicle_type="car").shape[0], 40)
        motor = A.plate_number_row(img, vehicle_type="motorcycle")
        self.assertEqual(motor.shape[0], 40)
        os.environ.pop("ANPR_CHAR_ROW_RATIO", None)
        os.environ.pop("ANPR_CHAR_ROW_RATIO_MOTOR", None)
        self.assertEqual(A.plate_number_row(img, vehicle_type="car").shape[0], 40)

        os.environ["ANPR_OCR_LONGER_WINS"] = "0"
        self.assertFalse(A.prefer_rapid_reading("B 1234 AB", 0.80, True, "B 1234 ABC", 0.40, True))
        self.assertTrue(A.prefer_rapid_reading("B 12 A", 0.40, False, "B 1234 ABC", 0.70, True))
        self.assertTrue(A.prefer_rapid_reading("B 1234 AB", 0.50, True, "B 1234 ABC", 0.80, True))
        os.environ["ANPR_OCR_LONGER_WINS"] = "1"
        self.assertTrue(A.prefer_rapid_reading("B 1234 AB", 0.80, True, "B 1234 ABC", 0.40, True))
        os.environ.pop("ANPR_OCR_LONGER_WINS", None)

    def test_12_cct_decode_and_can_be_disabled(self):
        """cct-xs membuang pad dan bisa dimatikan tanpa menyentuh model."""
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
        rows = []
        for ch in "B12":
            v = np.zeros(len(alphabet), dtype=np.float32)
            v[alphabet.index(ch)] = 1.0
            rows.append(v)
        for _ in range(7):
            v = np.zeros(len(alphabet), dtype=np.float32)
            v[alphabet.index("_")] = 1.0
            rows.append(v)
        text, conf = A.decode_cct_output(np.stack(rows), alphabet, "_")
        self.assertEqual(text, "B12")
        self.assertAlmostEqual(conf, 1.0)
        self.assertTrue(A.fast_plate_enabled())
        os.environ["ANPR_FAST_PLATE"] = "0"
        try:
            self.assertFalse(A.fast_plate_enabled())
        finally:
            os.environ.pop("ANPR_FAST_PLATE", None)

    def test_13_plate_format_spacing_and_compact(self):
        """Format plat nomor mendukung opsi gabung tanpa spasi (B1786CIC) dan spasi standar (B 1786 CIC)."""
        # 1. Format helper
        self.assertEqual(A.format_license_plate("B 1786 CIC", compact=True), "B1786CIC")
        self.assertEqual(A.format_license_plate("B1786CIC", compact=False), "B 1786 CIC")
        self.assertEqual(A.format_license_plate("523-07", compact=True), "523-07")
        self.assertEqual(A.format_license_plate("523-07", compact=False), "523-07")

        # 2. Refine Indonesian Plate dengan opsi compact
        self.assertEqual(A.refine_indonesian_plate("B1786CIC", compact=True), "B1786CIC")
        self.assertEqual(A.refine_indonesian_plate("B1786CIC", compact=False), "B 1786 CIC")

        # 3. Via Environment Variable ANPR_COMPACT_PLATE
        os.environ["ANPR_COMPACT_PLATE"] = "1"
        try:
            self.assertEqual(A.format_license_plate("B 1786 CIC"), "B1786CIC")
            self.assertEqual(A.refine_indonesian_plate("B 1786 CIC"), "B1786CIC")
        finally:
            os.environ.pop("ANPR_COMPACT_PLATE", None)

        os.environ["ANPR_COMPACT_PLATE"] = "0"
        try:
            self.assertEqual(A.format_license_plate("B1786CIC"), "B 1786 CIC")
            self.assertEqual(A.refine_indonesian_plate("B1786CIC"), "B 1786 CIC")
        finally:
            os.environ.pop("ANPR_COMPACT_PLATE", None)

    def test_14_alpr_kit_v29_contract_endpoints(self):
        """Pastikan 4 endpoint standar ALPR Kit v2.9 aktif dan merespon dengan benar."""
        client = A.app.test_client()

        # 1. /get_plate (POST & GET)
        dummy_rec = {
            "id": 123456,
            "license_plate": "B 1786 CIC",
            "vehicle_type": "car",
            "body_style": "mpv",
            "confidence": 0.93,
            "timestamp": "2026-11-13 18:00:00",
            "snapshot_url": "/captures/test_snapshot.jpg"
        }
        A.LATEST_RECORDS.appendleft(dummy_rec)

        res_post = client.post("/get_plate", json={})
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.get_json()
        self.assertEqual(data_post.get("status"), "success")
        self.assertIn("license_plate", data_post)

        res_get = client.get("/get_plate")
        self.assertEqual(res_get.status_code, 200)

        # 2. /get_image (POST & GET)
        test_img_path = os.path.join(A.CAPTURES_DIR, "test_dummy_image.jpg")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        A.cv2.imwrite(test_img_path, dummy_img)
        try:
            res_img = client.post("/get_image", json={"image_path": test_img_path})
            self.assertEqual(res_img.status_code, 200)
            self.assertEqual(res_img.mimetype, "image/jpeg")
            res_img.close()

            # 3. /delete_image (DELETE & POST)
            res_del = client.delete("/delete_image", json={"image_paths": [test_img_path]})
            self.assertEqual(res_del.status_code, 200)
            data_del = res_del.get_json()
            self.assertEqual(data_del.get("status"), "success")
            self.assertFalse(os.path.exists(test_img_path))
        finally:
            if os.path.exists(test_img_path):
                os.remove(test_img_path)
            if len(A.LATEST_RECORDS) > 0 and A.LATEST_RECORDS[0].get("id") == 123456:
                A.LATEST_RECORDS.popleft()


if __name__ == "__main__":
    unittest.main(verbosity=2)
