"""
=============================================================================
📺 RASPBERRY PI ANPR CCTV MONITOR CLIENT
=============================================================================
Script ini dijalankan di Raspberry Pi yang terhubung via kabel HDMI ke Monitor CCTV.

Kelebihan:
1. SANGAT RINGAN: Tidak perlu install PyTorch, Ultralytics, atau EasyOCR di RPi.
   Seluruh pemrosesan AI dijalankan di Laptop/Server.
2. TRUE REAL-TIME: Tampilan video bergerak mulus (25-30 FPS) di layar monitor.
3. DETEKSI INSTAN (<1 Detik): Mengambil hasil ANPR dari memory server tanpa jeda.

Cara Install Library di Raspberry Pi (Hanya butuh 2 library standar):
    sudo apt-get update
    sudo apt-get install -y python3-opencv python3-requests

Cara Menjalankan di Raspberry Pi:
    python3 rpi_cctv_monitor.py --server http://<IP_LAPTOP_ANDA>:5001

Tombol Kontrol di Layar Monitor:
    - [SPASI] : Scan manual seketika
    - [A]     : Nyalakan / Matikan Auto-Scan otomatis (setiap 2 detik)
    - [F]     : Toggle Layar Penuh (Fullscreen)
    - [Q / ESC]: Keluar
=============================================================================
"""

import cv2
import time
import requests
import argparse
import threading
from datetime import datetime

parser = argparse.ArgumentParser(description="Raspberry Pi CCTV Monitor Client for ANPR")
parser.add_argument("--server", default="http://127.0.0.1:5001", help="URL Server ANPR (contoh: http://192.168.1.100:5001)")
parser.add_argument("--fps", type=int, default=25, help="Target FPS tampilan monitor")
parser.add_argument("--auto", action="store_true", help="Langsung aktifkan Auto-Scan saat mulai")
args = parser.parse_args()

SERVER_URL = args.server.rstrip("/")
STREAM_URL = f"{SERVER_URL}/api/live_stream"
DETECT_URL = f"{SERVER_URL}/api/detect_current"

# State global
latest_detection = None
detection_lock = threading.Lock()
is_scanning = False
auto_scan = args.auto
last_scan_time = 0.0
scan_status_text = "Siap"
scan_time_sec = 0.0


def async_scan_worker():
    """Worker di thread latar belakang agar tampilan video di monitor tidak pernah freeze/patah-patah."""
    global is_scanning, latest_detection, scan_status_text, scan_time_sec
    try:
        t0 = time.time()
        resp = requests.post(DETECT_URL, timeout=4)
        scan_time_sec = round(time.time() - t0, 2)
        if resp.status_code == 200:
            data = resp.json()
            dets = data.get("detections", [])
            with detection_lock:
                if dets:
                    latest_detection = dets[0]
                    scan_status_text = f"Plat: {latest_detection.get('license_plate', '-')} ({scan_time_sec}s)"
                else:
                    latest_detection = None
                    scan_status_text = f"Tidak ada kendaraan ({scan_time_sec}s)"
        else:
            scan_status_text = f"Error Server ({resp.status_code})"
    except Exception as e:
        scan_status_text = f"Koneksi Putus: {str(e)[:20]}"
    finally:
        is_scanning = False


def trigger_scan():
    global is_scanning, scan_status_text
    if is_scanning:
        return
    is_scanning = True
    scan_status_text = "Memindai di RAM (<1s)..."
    threading.Thread(target=async_scan_worker, daemon=True).start()


def main():
    global auto_scan, last_scan_time

    print("=" * 65)
    print(f"🚀 Membuka ANPR CCTV Monitor Client ke Server: {SERVER_URL}")
    print(f"📹 Stream URL: {STREAM_URL}")
    print("Tekan 'Q' di jendela monitor untuk keluar, 'SPASI' untuk scan.")
    print("=" * 65)

    window_name = "ANPR Smart CCTV Monitor - Gate 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka stream dari {STREAM_URL}")
        print("Pastikan app.py di laptop sudah berjalan dan kamera sudah Connect.")
        return

    fps_display = 0.0
    frame_count = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_display = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # Auto-scan terjadwal setiap 2 detik
        if auto_scan and (now - last_scan_time >= 2.0) and not is_scanning:
            last_scan_time = now
            trigger_scan()

        # Salin deteksi aktif dengan thread-safe
        with detection_lock:
            det = latest_detection

        # Gambar Bounding Box Kendaraan (Biru)
        if det and det.get("bbox"):
            x1, y1, x2, y2 = det["bbox"]
            v_type = det.get("vehicle_type", "Vehicle").title()
            b_style = det.get("body_style", "")
            lbl = f"{v_type}" + (f" - {b_style}" if b_style and b_style.lower() != v_type.lower() else "")
            
            # Box mobil
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + len(lbl)*11 + 10, y1), (255, 120, 0), -1)
            cv2.putText(frame, lbl, (x1 + 6, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Gambar Bounding Box Plat Nomor (Kuning)
        if det and det.get("plate_bbox"):
            px1, py1, px2, py2 = det["plate_bbox"]
            plate_str = det.get("license_plate") or "PLAT"
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 215, 255), 3)
            p_label = f"[{plate_str}]"
            cv2.rectangle(frame, (px1, max(0, py1 - 28)), (px1 + len(p_label)*13 + 10, py1), (0, 215, 255), -1)
            cv2.putText(frame, p_label, (px1 + 6, max(18, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        # Header OSD CCTV Monitor
        cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        osd_left = f"GATE 01 - LIVE | {time_str} | {fps_display:.1f} FPS"
        cv2.putText(frame, osd_left, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        auto_lbl = "[AUTO-SCAN ON]" if auto_scan else "[MANUAL SCAN]"
        osd_right = f"{auto_lbl} | Status: {scan_status_text}"
        tw = cv2.getTextSize(osd_right, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.putText(frame, osd_right, (w - tw - 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27): # Q or ESC
            break
        elif key == ord(' '): # Spacebar
            trigger_scan()
        elif key in (ord('a'), ord('A')):
            auto_scan = not auto_scan
            print(f"Auto-scan diubah ke: {auto_scan}")
        elif key in (ord('f'), ord('F')):
            # Toggle fullscreen
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            new_prop = cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_prop)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
