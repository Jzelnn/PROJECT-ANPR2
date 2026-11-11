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


if __name__ == "__main__":
    unittest.main(verbosity=2)
