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


if __name__ == "__main__":
    unittest.main(verbosity=2)
