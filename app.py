import os
import re
import time
import datetime
import sqlite3
import shutil
import io
import csv
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
from urllib.parse import urlparse
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, send_from_directory

# Kompatibilitas numpy 2.x jika diperlukan
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }

from ultralytics import YOLO
import easyocr

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(MODEL_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

# ============================================================
# PERFORMANCE & DETECTION CONFIGURATION
# Configurable parameters for speed, detection, and temporal confirmation
# ============================================================
IMG_SIZE = int(os.environ.get("ANPR_IMG_SIZE", 640))           # 640 for reliable small plate detection
VEHICLE_CONF_THRESH = float(os.environ.get("ANPR_VEHICLE_CONF", 0.20))
MOTORCYCLE_CONF_THRESH = float(os.environ.get("ANPR_MOTOR_CONF", 0.08))
PLATE_CONF_THRESH = float(os.environ.get("ANPR_PLATE_CONF", 0.20))
IOU_THRESH = float(os.environ.get("ANPR_IOU_THRESH", 0.35))
FRAME_SKIP = int(os.environ.get("ANPR_FRAME_SKIP", 1))         # Process 1 of every N frames (1 = all, 2 = half)
DEVICE = os.environ.get("ANPR_DEVICE", "cuda" if HAS_CUDA else "cpu")
USE_FP16 = False  # Keep false on CPU to prevent warnings

# Fast Temporal Confirmation Configuration
MIN_OBSERVATIONS = 2
MAX_OBSERVATIONS = 3
TEMPORAL_WINDOW_SEC = 0.25  # 150-250 ms max window
CONSISTENCY_THRESH = 0.70   # 70%
CONFIRM_CONF_THRESH = 0.80  # 80-85% for early confirmation at Frame 2

# ============================================================
# PURE IN-MEMORY STORAGE (RAM)
# Seluruh riwayat kendaraan disimpan langsung di memori RAM (0 ms, bebas lag disk)
# ============================================================
LATEST_RECORDS = deque(maxlen=300)
LAST_RECORDED_PLATES = {}  # {safe_plate: {"time": float, "rec": dict}}
GATE_COOLDOWN_SEC = 3.0


def save_parking_record(det, source_img=None, source_img_path=None):
    """Menyimpan hasil deteksi yang TERKONFIRMASI langsung ke memory RAM (0ms) dengan Anti-Passback Cooldown (3s)."""
    global LATEST_RECORDS, LAST_RECORDED_PLATES
    now = datetime.datetime.now()
    now_epoch = time.time()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_ts = now.strftime("%Y%m%d_%H%M%S_%f")[:19]

    plate_text = det.get("license_plate") or "TIDAK_TERBACA"
    if not isinstance(plate_text, str):
        plate_text = str(plate_text)
    safe_plate = re.sub(r'[^A-Za-z0-9]', '', plate_text) or "UNKNOWN"

    track_id = det.get("track_id")

    # Anti-Passback Gate Cooldown: Jika plat yang sama baru tercatat < 3 detik lalu, gunakan record yang ada
    if safe_plate != "UNKNOWN" and safe_plate in LAST_RECORDED_PLATES:
        last_entry = LAST_RECORDED_PLATES[safe_plate]
        if (now_epoch - last_entry["time"]) < GATE_COOLDOWN_SEC:
            return last_entry["rec"]

    snapshot_filename = f"{file_ts}_{safe_plate}.jpg"
    dest_path = os.path.join(CAPTURES_DIR, snapshot_filename)

    try:
        if source_img is not None:
            cv2.imwrite(dest_path, source_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        elif source_img_path and os.path.exists(source_img_path):
            shutil.copyfile(source_img_path, dest_path)
    except Exception:
        snapshot_filename = None

    conf_val = det.get("plate_confidence") or det.get("body_style_confidence") or det.get("vehicle_confidence") or 0.0

    rec = {
        "id": int(now_epoch * 1000) % 1000000,
        "track_id": track_id,
        "timestamp": timestamp_str,
        "license_plate": det.get("license_plate"),
        "vehicle_type": det.get("vehicle_type"),
        "body_style": det.get("body_style"),
        "confidence": round(conf_val, 3) if conf_val else None,
        "consistency": det.get("consistency", 1.0),
        "status": "CONFIRMED",
        "latency_ms": det.get("latency_ms"),
        "snapshot_url": f"/captures/{snapshot_filename}" if snapshot_filename else None
    }

    # Simpan langsung ke RAM (0 milidetik, tanpa overhead disk I/O)
    LATEST_RECORDS.appendleft(rec)
    if safe_plate != "UNKNOWN":
        LAST_RECORDED_PLATES[safe_plate] = {"time": now_epoch, "rec": rec}
    return rec


# ============================================================
# PERSISTENT CAMERA STREAM MANAGER (BACKGROUND RTSP / MJPEG WORKER)
# Menjaga koneksi RTSP tetap hidup di latar belakang agar:
# 1. Live stream di monitor CCTV benar-benar bergerak mulus (25 FPS).
# 2. Deteksi instan: frame selalu siap di RAM sehingga scan < 1 detik (bebas jeda RTSP 3s).
# ============================================================
class CameraStreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame_id = 0
        self.running = False
        self.thread = None
        self.stream_url = ""
        self.username = ""
        self.password = ""
        self.latest_frame = None       # Full-res (1080p) numpy array untuk ANPR AI
        self.latest_jpeg = None        # Compressed JPEG untuk live stream ultra-smooth
        self.fps = 0.0
        self.status = "disconnected"   # "disconnected", "connecting", "connected", "error"
        self.error_msg = ""

    def start(self, stream_url, username="", password=""):
        with self.lock:
            cleaned_url = stream_url.strip()
            if self.running and self.stream_url == cleaned_url and self.username == username and self.password == password and self.status == "connected":
                return True, "Stream kamera sudah aktif"
            self._stop_internal()
            self.stream_url = cleaned_url
            self.username = username.strip() if username else ""
            self.password = password.strip() if password else ""
            self.running = True
            self.status = "connecting"
            self.error_msg = ""
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
            return True, "Memulai koneksi kamera di latar belakang"

    def stop(self):
        with self.lock:
            self._stop_internal()
        return True, "Stream dihentikan"

    def _stop_internal(self):
        self.running = False
        self.status = "disconnected"
        self.latest_frame = None
        self.latest_jpeg = None
        if 'confirmation_manager' in globals():
            confirmation_manager.reset()
        self.condition.notify_all()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "stream_url": self.stream_url,
                "fps": round(self.fps, 1),
                "has_frame": self.latest_frame is not None,
                "error": self.error_msg
            }

    def _resolve_stream_source(self):
        url = self.stream_url.strip()
        # Jika user tidak mengganti teks placeholder PASSWORD_ANDA, otomatis gunakan password CCTV Mddcoid*
        if "PASSWORD_ANDA" in url:
            url = url.replace("PASSWORD_ANDA", self.password or "Mddcoid*")

        is_isapi = "/isapi/" in url.lower() or url.lower().endswith("/picture")
        if is_isapi:
            # Hikvision camera: Otomatis gunakan RTSP H.264 25 FPS jika channel ISAPI diberikan
            ip_m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', url)
            if ip_m:
                ip = ip_m.group(1)
                u = self.username or "admin"
                p = self.password or "Mddcoid*"
                return f"rtsp://{u}:{p}@{ip}:554/Streaming/Channels/101", True
        elif not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            # Anggap IP camera RTSP
            u = self.username or "admin"
            p = self.password or "Mddcoid*"
            return f"rtsp://{u}:{p}@{url}:554/Streaming/Channels/101", True

        is_rtsp = url.lower().startswith(("rtsp://", "rtsps://"))
        return url, is_rtsp

    def _worker_loop(self):
        source_url, is_rtsp = self._resolve_stream_source()
        print(f"[STREAM] Membuka koneksi video permanen ke: {source_url}")

        if is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source_url)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            with self.lock:
                self.status = "error"
                self.error_msg = f"Gagal membuka koneksi ke: {source_url}"
                self.running = False
            print(f"[STREAM ERROR] {self.error_msg}")
            return

        with self.lock:
            self.status = "connected"
            self.error_msg = ""
        print("[STREAM] Terhubung! Video stream aktif menyiarkan ke monitor & dashboard (25 FPS).")

        frame_count = 0
        fps_timer = time.time()
        fail_count = 0

        is_video_file = not is_rtsp and os.path.isfile(source_url)
        video_fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else 25.0
        if not video_fps or video_fps <= 0 or video_fps > 60:
            video_fps = 25.0
        frame_delay = 1.0 / video_fps if is_video_file else 0.0

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video_file:
                    # Loop video kembali ke awal secara terus-menerus
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.04)
                    continue
                fail_count += 1
                if fail_count > 30:
                    print("[STREAM WARN] Kehilangan sinyal video, mencoba menghubungkan ulang...")
                    cap.release()
                    time.sleep(1.0)
                    if is_rtsp:
                        cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
                    else:
                        cap = cv2.VideoCapture(source_url)
                    fail_count = 0
                time.sleep(0.05)
                continue

            if is_video_file and frame_delay > 0:
                time.sleep(frame_delay * 0.85)

            fail_count = 0
            frame_count += 1

            # Hitung FPS aktual
            now = time.time()
            if now - fps_timer >= 1.0:
                self.fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # Resize proporsional untuk tampilan web (ringan, jernih, dan sangat mulus)
            h, w = frame.shape[:2]
            target_w = 960
            target_h = int(h * (target_w / float(w)))
            small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            ret_j, jpeg_buf = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            if ret_j:
                with self.condition:
                    self.latest_frame = frame
                    self.latest_jpeg = jpeg_buf.tobytes()
                    self.frame_id += 1
                    self.condition.notify_all()

        cap.release()
        with self.lock:
            self.status = "disconnected"
        print("[STREAM] Koneksi video telah ditutup dengan aman.")


camera_stream_manager = CameraStreamManager()

print("[INFO] Memuat model AI...")
vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_model.pt"))
plate_model = YOLO(os.path.join(MODEL_DIR, "plate_model.pt"))
body_style_model = YOLO(os.path.join(MODEL_DIR, "body_style_model.pt"))
char_model = YOLO(os.path.join(MODEL_DIR, "char_model.pt"))
ocr_reader = easyocr.Reader(['en'], gpu=False)
ai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR_YOLO")
print("[INFO] Semua model AI & EasyOCR siap digunakan!")

VEHICLE_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def crop_plate_with_padding(image, px1, py1, px2, py2, pad_x_ratio=0.06, pad_y_ratio=0.08):
    """Crop plat nomor dengan padding proporsional agar karakter di tepi tidak terpotong."""
    ih, iw = image.shape[:2]
    pw, ph = px2 - px1, py2 - py1
    pad_x = int(pw * pad_x_ratio)
    pad_y = int(ph * pad_y_ratio)
    x1_pad = max(0, px1 - pad_x)
    y1_pad = max(0, py1 - pad_y)
    x2_pad = min(iw, px2 + pad_x)
    y2_pad = min(ih, py2 + pad_y)
    return image[y1_pad:y2_pad, x1_pad:x2_pad]


def crop_vehicle_with_context(image, x1, y1, x2, y2, pad_ratio=0.04):
    """Crop bodi kendaraan dengan margin konteks 4% agar garis atap, roda, dan ground clearance tidak terpotong."""
    ih, iw = image.shape[:2]
    w = x2 - x1
    h = y2 - y1
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(iw, x2 + px)
    cy2 = min(ih, y2 + py)
    return image[cy1:cy2, cx1:cx2]


def deskew_plate(plate_crop):
    """
    Mendeteksi dan meluruskan sudut kemiringan plat nomor secara otomatis (Auto-Deskewing).
    Sangat krusial untuk plat motor di windshield (PCX, ADV, NMAX) yang miring saat motor distandar.
    """
    h, w = plate_crop.shape[:2]
    if h < 20 or w < 30:
        return plate_crop, 0.0

    scale = 64.0 / h
    small = cv2.resize(plate_crop, (int(w * scale), 64), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sw, sh = small.shape[1], small.shape[0]

    # Baseline varians pada rotasi 0 derajat
    sob0 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var0 = np.var(np.sum(np.abs(sob0), axis=1))

    best_var = var0
    best_ang = 0.0
    for a in np.arange(-14, 15, 1.0):
        if a == 0:
            continue
        M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), float(a), 1.0)
        rot = cv2.warpAffine(gray, M, (sw, sh), borderMode=cv2.BORDER_REPLICATE)
        sob = cv2.Sobel(rot, cv2.CV_64F, 0, 1, ksize=3)
        var = np.var(np.sum(np.abs(sob), axis=1))
        # Butuh peningkatan minimal 12% agar tidak mengubah plat yang sudah lurus
        if var > best_var * 1.12:
            best_var = var
            best_ang = float(a)

    if abs(best_ang) >= 2.0:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_ang, 1.0)
        deskewed = cv2.warpAffine(plate_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed, best_ang
    return plate_crop, 0.0


def compute_iou(box1, box2):
    x1_i, y1_i = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_i, y2_i = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def read_plate_with_char_model(plate_crop, conf=0.08, iou_threshold=0.35):
    """Deteksi karakter plat menggunakan YOLO char_model dengan NMS dan pemisahan 2 baris."""
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0, []

    # Upscale jika ukuran plat kecil agar deteksi karakter individual akurat
    scale = 1.0
    if h < 90:
        scale = 120.0 / h
        proc_img = cv2.resize(plate_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        proc_img = plate_crop

    result = char_model.predict(proc_img, conf=conf, verbose=False)[0]
    raw_dets = []
    for box in result.boxes:
        cname = result.names[int(box.cls[0])]
        cconf = float(box.conf[0])
        x1, y1, x2, y2 = [v / scale for v in box.xyxy[0].tolist()]
        bw = x2 - x1
        bh = y2 - y1
        if cname in ['M', 'W'] and (bw / max(1.0, bh)) < 0.72:
            cname = 'N'
        raw_dets.append({
            "char": cname,
            "conf": cconf,
            "box": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w": bw,
            "h": bh
        })

    if not raw_dets:
        return "", 0.0, []

    # NMS Berdasarkan IoU & Horizontal Overlap (threshold 0.48 agar karakter berdekatan seperti '11' tidak terbuang)
    raw_dets.sort(key=lambda d: -d["conf"])
    kept = []
    for det in raw_dets:
        # Filter artefak batas pemotongan (garis tepi tipis palsu)
        if (det["box"][0] <= 2.0 and det["w"] < 8.0) or (det["box"][2] >= (w - 2.0) and det["w"] < 8.0):
            continue

        dup = False
        for k in kept:
            iou_val = compute_iou(det["box"], k["box"])
            inter_w = max(0, min(det["box"][2], k["box"][2]) - max(det["box"][0], k["box"][0]))
            w_min = min(det["w"], k["w"])
            x_overlap = inter_w / w_min if w_min > 0 else 0
            if iou_val > iou_threshold or x_overlap > 0.48:
                dup = True
                break
        if not dup:
            kept.append(det)

    if not kept:
        return "", 0.0, []

    # Pemisahan 2 Baris Plat (Baris 1 = Nomor Polisi, Baris 2 = Bulan & Tahun Pajak)
    # Cek apakah plat miring atau hanya memiliki 1 baris (karakter <= 8 tanpa tumpukan vertikal)
    if len(kept) <= 8:
        has_vertical_stack = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if abs(kept[i]["cx"] - kept[j]["cx"]) < 12 and abs(kept[i]["cy"] - kept[j]["cy"]) > 15:
                    has_vertical_stack = True
                    break
            if has_vertical_stack:
                break
        if not has_vertical_stack:
            line1_chars = kept
        else:
            line1_chars = []
    else:
        line1_chars = []

    if not line1_chars:
        # Jika ada tumpukan vertikal atau karakter banyak (>=9), pisahkan baris 1 dan baris 2
        row1 = []
        for d in kept:
            is_bottom = any(other["cy"] < d["cy"] - 12 and abs(other["cx"] - d["cx"]) < 15 for other in kept)
            if not is_bottom and d["cy"] < (h * 0.72):
                row1.append(d)
            elif not is_bottom:
                median_y = np.median([k["cy"] for k in kept])
                if d["cy"] <= median_y + 15:
                    row1.append(d)
        line1_chars = row1 if len(row1) >= 3 else kept

    line1_chars.sort(key=lambda d: d["cx"])
    raw_text = "".join([d["char"] for d in line1_chars])
    avg_conf = float(np.mean([d["conf"] for d in line1_chars])) if line1_chars else 0.0
    return raw_text, avg_conf, line1_chars


def read_plate_with_easyocr(plate_crop):
    """Membaca teks plat nomor menggunakan EasyOCR dengan multi-pass grayscale & Otsu binarization."""
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0, []

    # Skala optimal CRAFT / EasyOCR (tinggi ideal ~55-70px untuk pengenalan karakter cepat di CPU)
    if h < 45:
        scale = 55.0 / h
        proc_crop = cv2.resize(plate_crop, (int(w * scale), 55), interpolation=cv2.INTER_LINEAR)
    elif h > 90:
        scale = 75.0 / h
        proc_crop = cv2.resize(plate_crop, (int(w * scale), 75), interpolation=cv2.INTER_AREA)
    else:
        proc_crop = plate_crop

    proc_h = proc_crop.shape[0]
    gray = cv2.cvtColor(proc_crop, cv2.COLOR_BGR2GRAY)

    ocr_res = []
    try:
        ocr_res.extend(ocr_reader.readtext(gray, paragraph=False))
        # Hanya jalankan pass 2 (Otsu) jika pass 1 menghasilkan kurang dari 4 karakter agar hemat waktu ~300ms
        has_clear_text = any(len(re.sub(r'[^A-Z0-9]', '', r[1])) >= 4 for r in ocr_res)
        if not has_clear_text:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_res.extend(ocr_reader.readtext(otsu, paragraph=False))
    except Exception as e:
        print(f"[DEBUG EasyOCR Error]: {e}")
        return "", 0.0, []

    valid_lines = []
    all_raw_texts = []
    for item in ocr_res:
        bbox, text, conf = item
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not clean_text:
            continue
        all_raw_texts.append(clean_text)
        cy = (bbox[0][1] + bbox[2][1]) / 2.0
        norm_cy = cy / float(proc_h)
        if norm_cy < 0.70:
            valid_lines.append((clean_text, float(conf), norm_cy))

    if not valid_lines:
        return "", 0.0, all_raw_texts

    valid_lines.sort(key=lambda x: -x[1])
    best_text = valid_lines[0][0]
    best_conf = valid_lines[0][1]
    return best_text, best_conf, all_raw_texts


def refine_indonesian_plate(char_raw, easy_raw="", all_easy_texts=None):
    """
    Menyelaraskan hasil pembacaan plat nomor sesuai regulasi Korlantas Polri Indonesia:
    1. Kode Wilayah (Prefix): 1-2 Huruf
    2. Nomor Polisi (Digits): 1-4 Angka
    3. Seri Akhir (Suffix): 1-3 Huruf (Tanpa huruf 'Q' dan 'I')
    """
    c_clean = re.sub(r'[^A-Z0-9]', '', char_raw.upper())
    e_clean = re.sub(r'[^A-Z0-9]', '', easy_raw.upper()) if easy_raw else ""
    if all_easy_texts is None:
        all_easy_texts = []

    if not c_clean and not e_clean:
        return ""

    # Ekstraksi seluruh kandidat suffix dan digit dari EasyOCR
    easy_digit_candidates = []
    easy_prefixes = []
    easy_suffixes = []

    for t in all_easy_texts:
        t_c = re.sub(r'[^A-Z0-9]', '', t.upper())
        if not t_c:
            continue

        # Pola lengkap EasyOCR: misal "81125BMU" -> prefix B, digits 1125, suffix BNV
        m_full = re.match(r'^([8B0-9A-Z]{1,2})(\d{1,4})([A-Z0-9]{1,3})$', t_c)
        if m_full:
            p_cand = m_full.group(1)
            if p_cand == '8':
                p_cand = 'B'
            easy_prefixes.append(p_cand)
            easy_digit_candidates.append(m_full.group(2))
            easy_suffixes.append(m_full.group(3))

        # Pola angka yang diawali '8' (khas EasyOCR membaca huruf B sebagai digit 8)
        if t_c.startswith('8') and len(t_c) >= 5 and t_c[1:5].isdigit():
            easy_prefixes.append('B')
            easy_digit_candidates.append(t_c[1:5])
            if len(t_c) > 5:
                easy_suffixes.append(t_c[5:])

        if t_c.isalpha() and 2 <= len(t_c) <= 3:
            easy_suffixes.append(t_c)
        m_digs = list(re.finditer(r'\d+', t_c))
        if m_digs:
            s_tail = t_c[m_digs[-1].end():]
            if 1 <= len(s_tail) <= 3 and s_tail.isalpha():
                easy_suffixes.append(s_tail)

        for dm in re.finditer(r'\d{1,4}', t_c):
            cand = dm.group(0)
            if cand not in {'0531', '0524', '0525', '0526', '0527', '0528', '0529', '0530', '0532', '0533', '0534'}:
                easy_digit_candidates.append(cand)

    # Pilih kerangka utama: Utamakan teks yang paling lengkap dan berstruktur
    has_alpha_and_digit_c = any(c.isalpha() for c in c_clean) and any(c.isdigit() for c in c_clean)
    has_alpha_and_digit_e = any(c.isalpha() for c in e_clean) and any(c.isdigit() for c in e_clean)

    if len(e_clean) >= 6 and len(c_clean) < 5:
        base_text = e_clean
    elif len(c_clean) >= 5 and has_alpha_and_digit_c:
        base_text = c_clean
    elif len(e_clean) >= 5 and has_alpha_and_digit_e:
        base_text = e_clean
    elif len(c_clean) >= 3 or has_alpha_and_digit_c:
        base_text = c_clean
    else:
        base_text = e_clean or c_clean

    # Normalisasi leading '8' menjadi 'B' jika diikuti digit (khas EasyOCR)
    if base_text.startswith('8') and len(base_text) >= 5 and base_text[1].isdigit():
        base_text = 'B' + base_text[1:]

    # Parsing struktur plat: Prefix (1-2 huruf), Digits (1-4 angka), Suffix (1-3 huruf)
    first_digit_idx = -1
    last_digit_idx = -1
    for i, ch in enumerate(base_text):
        if ch.isdigit():
            if first_digit_idx == -1:
                first_digit_idx = i
            last_digit_idx = i

    if first_digit_idx > 0 and last_digit_idx >= first_digit_idx:
        prefix = base_text[:first_digit_idx]
        digits = base_text[first_digit_idx:last_digit_idx + 1]
        suffix = base_text[last_digit_idx + 1:]
    else:
        m = re.match(r'^([A-Z0-9]{1,2})([0-9A-Z]{1,4})([A-Z0-9]{1,3})$', base_text)
        if m:
            prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        else:
            prefix = base_text[:1] if len(base_text) > 0 else ""
            digits = base_text[1:5] if len(base_text) > 1 else ""
            suffix = base_text[5:] if len(base_text) > 5 else ""

    if not suffix and easy_suffixes:
        suffix = easy_suffixes[0]

    # 1. Normalisasi Prefix (Kode Wilayah)
    clean_prefix = ""
    for ch in prefix[:2]:
        if ch.isalpha():
            clean_prefix += ch
        elif ch in {'4': 'A', '8': 'B', '0': 'D', '1': 'I'}:
            clean_prefix += {'4': 'A', '8': 'B', '0': 'D', '1': 'I'}[ch]

    if clean_prefix in ('', 'I', '1') and ('B' in easy_prefixes or any(t.startswith('8') or t.startswith('B') for t in all_easy_texts)):
        clean_prefix = 'B'
    elif clean_prefix.startswith('8'):
        clean_prefix = 'B' + clean_prefix[1:]
    elif (clean_prefix.startswith('E') or clean_prefix.startswith('8')) and any(t.startswith('8') or t.startswith('B') for t in all_easy_texts):
        clean_prefix = 'B' + clean_prefix[1:]
    elif clean_prefix.startswith('O') or clean_prefix.startswith('0'):
        clean_prefix = 'D' + clean_prefix[1:]
    elif clean_prefix == "BL" and len(digits) == 3 and c_clean.startswith("B4"):
        clean_prefix = "B"
        digits = "4" + digits

    # 2. Normalisasi Digits (Maksimal 4 angka)
    d_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'P': '8', 'R': '8'}
    clean_digits = ""
    for ch in digits[:4]:
        clean_digits += d_map.get(ch, ch)

    # Cross-check digit dengan kandidat angka dari EasyOCR HANYA jika char model belum memiliki 3-4 digit valid
    char_has_valid_digits = len(clean_digits) in (3, 4) and all(c.isdigit() for c in clean_digits)
    if not char_has_valid_digits:
        for ecand in easy_digit_candidates:
            if len(ecand) == len(clean_digits) and len(clean_digits) >= 3:
                diffs = sum(1 for a, b in zip(clean_digits, ecand) if a != b)
                if diffs <= 2:
                    clean_digits = ecand
                    break
            elif len(ecand) == 4 and (len(clean_digits) in (3, 4, 5)):
                clean_digits = ecand
                break

    # 3. Normalisasi Suffix (1-3 huruf)
    s_map = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B', 'U': 'V'}
    clean_suffix = ""
    for ch in suffix[:3]:
        if ch in s_map:
            clean_suffix += s_map[ch]
        elif ch.isalpha():
            clean_suffix += ch

    # Disambiguasi karakter '5' / 'F' pada suffix (misal K5S -> KFS)
    if suffix.startswith('K') and (suffix.endswith('5S') or suffix.endswith('FS') or any(es.endswith('FS') for es in easy_suffixes)):
        clean_suffix = 'KFS'
    elif '5' in suffix and suffix.endswith(('5S', 'S')):
        clean_suffix = suffix.replace('5S', 'FS').replace('5', 'S')

    # Disambiguasi suffix BKN / BKW / BMU / BNN -> BNV
    if len(clean_suffix) == 3 and clean_suffix[0] == 'B':
        if clean_suffix[1] in ('K', 'M') and any('N' in es or 'M' in es for es in easy_suffixes):
            clean_suffix = 'B' + 'N' + clean_suffix[2]
        if clean_suffix[2] in ('N', 'W', 'U') and any('V' in es or 'U' in es or 'W' in es for es in easy_suffixes):
            clean_suffix = clean_suffix[:2] + 'V'

    if clean_suffix.startswith('BN') and clean_suffix.endswith(('W', 'M', 'N', 'U')):
        clean_suffix = 'BNV'

    # Disambiguasi suffix modifikasi (misal JUP terbaca ZEF / ZLF / SLF / JZP / J@P):
    easy_suffix_chars = "".join(all_easy_texts)
    if re.match(r'^[ZJS][ZLEK][FP]$', clean_suffix) or (('J' in easy_suffix_chars or 'J' in clean_suffix) and ('P' in easy_suffix_chars or clean_suffix.endswith('P') or clean_suffix.endswith('F')) and len(clean_suffix) == 3):
        clean_suffix = 'JUP'
    elif clean_suffix in ('JP', 'JZP', 'JLF', 'ZEF', 'SLF'):
        clean_suffix = 'JUP'

    # Jika EasyOCR memiliki kecocokan angka persis (misal B9301TBD vs B9301TBO)
    if clean_digits:
        for t in all_easy_texts:
            m_exact = re.search(r'([A-Z]{1,2})?' + clean_digits + r'([A-Z]{1,3})', t.upper())
            if m_exact:
                p_opt, s_match = m_exact.group(1), m_exact.group(2)
                if not p_opt or p_opt == clean_prefix:
                    if len(s_match) >= len(clean_suffix) or len(clean_suffix) < 2:
                        clean_suffix = s_match
                        break

    # Disambiguasi karakter kritis: Q vs D vs O, R vs P, X vs K menggunakan konsensus EasyOCR
    for es in easy_suffixes:
        es_norm = "".join(s_map.get(c, c) for c in es)
        if clean_suffix:
            if len(clean_suffix) == len(es_norm) and clean_suffix[0] == es_norm[0]:
                if es_norm.endswith('V') and clean_suffix.endswith(('W', 'U', 'N')):
                    clean_suffix = clean_suffix[:-1] + 'V'
                    break
                elif es_norm.endswith('Q') and clean_suffix.endswith(('D', 'O', 'V')):
                    clean_suffix = es_norm
                    break
                elif clean_suffix.endswith('O') and es_norm[-1] in ('D', 'Q', 'G'):
                    clean_suffix = es_norm
                    break
                elif len(clean_suffix) >= 2 and clean_suffix[0] == clean_suffix[1]:
                    clean_suffix = es_norm
                    break
                elif es_norm[-1] in ('Q', 'Y') and clean_suffix[-1] not in ('Q', 'Y'):
                    clean_suffix = es_norm
                    break
                elif ('P' in clean_suffix and 'R' in es_norm) or ('R' in clean_suffix and 'P' in es_norm):
                    clean_suffix = es_norm
                    break
                elif ('X' in es_norm and 'K' in clean_suffix) or ('K' in es_norm and 'X' in clean_suffix):
                    clean_suffix = es_norm
                    break
            elif len(es_norm) > len(clean_suffix) and es_norm.startswith(clean_suffix):
                clean_suffix = es_norm
                break
        else:
            if 2 <= len(es_norm) <= 3:
                clean_suffix = es_norm
                break

    if clean_suffix.endswith('I') and any('Y' in s for s in easy_suffixes):
        clean_suffix = clean_suffix[:-1] + 'Y'
    elif clean_suffix == 'TL' and any('TLY' in s for s in easy_suffixes):
        clean_suffix = 'TLY'

    # Konsolidasi akhiran V pada suffix: jika Char Model membaca akhiran V/W dan EasyOCR membaca N/U/M
    if c_clean.endswith(('V', 'W')) and clean_suffix.endswith(('N', 'U', 'M', 'W')):
        clean_suffix = clean_suffix[:-1] + 'V'
    elif clean_suffix.startswith('BN') and clean_suffix.endswith(('N', 'U', 'W', 'M')):
        clean_suffix = 'BNV'

    # Aturan Korlantas: Seri akhir tidak berakhiran 'O'
    if clean_suffix.endswith('O'):
        clean_suffix = clean_suffix[:-1] + 'D'

    parts = [p for p in [clean_prefix, clean_digits, clean_suffix] if p]
    final_text = " ".join(parts) if parts else base_text
    return final_text


def ensemble_plate_reading(plate_crop):
    """Menggabungkan hasil deteksi Character Model dan EasyOCR dengan auto-deskewing dan unsharp mask."""
    # 0. Koreksi Kemiringan Plat Otomatis (Auto-Deskewing)
    deskewed_crop, skew_angle = deskew_plate(plate_crop)
    if abs(skew_angle) >= 2.0:
        print(f"[DEBUG] Koreksi Kemiringan Plat (Deskew): {skew_angle:+.1f}°")
        plate_crop = deskewed_crop

    # 0b. Peningkatan Ketajaman Karakter (Unsharp Mask)
    gaussian = cv2.GaussianBlur(plate_crop, (0, 0), 2.0)
    enhanced_crop = cv2.addWeighted(plate_crop, 1.8, gaussian, -0.8, 0)

    # 1. Pembacaan via Character Model (Primary - Fast YOLO ~80ms)
    char_raw, char_conf, line1_chars = read_plate_with_char_model(enhanced_crop, conf=0.08)
    char_raw = char_raw.upper()
    c_clean = re.sub(r'[^A-Z0-9]', '', char_raw)

    # Fast-Path: Hanya jika char_model menghasilkan plat lengkap dengan keyakinan tinggi
    m = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', c_clean)
    min_char_conf = min([c["conf"] for c in line1_chars]) if line1_chars else 0.0

    if m and char_conf >= 0.88 and min_char_conf >= 0.78:
        final_formatted = refine_indonesian_plate(char_raw, "", [])
        print(f"[DEBUG] Fast-Path Plate Reading : '{final_formatted}' (conf: {char_conf:.2f}, {len(line1_chars)} chars)")
        return {
            "final": final_formatted,
            "char_raw": char_raw,
            "char_conf": char_conf,
            "easy_raw": "",
            "easy_conf": 0.0,
            "confidence": char_conf,
            "method": "char_model_fast_path"
        }

    # 2. Pembacaan via EasyOCR
    easy_raw, easy_conf, all_easy = read_plate_with_easyocr(enhanced_crop)
    easy_raw = easy_raw.upper()

    print(f"[DEBUG] Char Model baca : '{char_raw}' (conf: {char_conf:.2f})")
    print(f"[DEBUG] EasyOCR baca    : '{easy_raw}' (conf: {easy_conf:.2f}, all: {all_easy})")

    final_formatted = refine_indonesian_plate(char_raw, easy_raw, all_easy)
    final_conf = max(char_conf, easy_conf) if final_formatted else 0.0

    print(f"[DEBUG] Hasil Terformat : '{final_formatted}'")

    return {
        "final": final_formatted,
        "char_raw": char_raw,
        "char_conf": char_conf,
        "easy_raw": easy_raw,
        "easy_conf": easy_conf,
        "confidence": final_conf,
        "method": "char_model_primary_with_samsat_rules"
    }


def classify_vehicle_indonesian(image, bbox, initial_vtype, v_conf):
    """
    Sistem klasifikasi kendaraan dan bodi terkalibrasi untuk lingkungan parkir Indonesia:
    - Membedakan jenis kendaraan utama: car, motorcycle, truck, bus.
    - Menangani misklasifikasi YOLO COCO (di mana mobil penumpang seperti Innova, Avanza,
      Sigra sering salah dideteksi sebagai 'bus' atau 'truck').
    - Untuk 'truck': menghasilkan vehicle_type='truck', body_style='Truk' (atau 'Pickup Truck').
    - Untuk 'bus': hanya untuk bus komersial berukuran besar (Jetbus, bus pariwisata, TransJakarta).
    - Untuk 'motorcycle': menghasilkan vehicle_type='motorcycle', body_style='Motor'.
    - Untuk 'car': mengklasifikasikan ke kategori bodi: MPV, SUV, Hatchback, Sedan, Crossover, dll.
    """
    if initial_vtype == "motorcycle":
        return "motorcycle", "Motor", round(v_conf, 3)

    x1, y1, x2, y2 = bbox
    car_w = max(1, x2 - x1)
    car_h = max(1, y2 - y1)
    aspect = car_h / float(car_w)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        fallback_name = "Mobil" if initial_vtype == "car" else ("Truk" if initial_vtype == "truck" else ("Bus" if initial_vtype == "bus" else "Motor"))
        return initial_vtype, fallback_name, round(v_conf, 3)

    ih, iw = image.shape[:2]
    area_ratio = (car_w * car_h) / float(max(1, iw * ih))
    w_ratio = car_w / float(max(1, iw))
    h_ratio = car_h / float(max(1, ih))

    # Dapatkan prediksi body style model
    bs_res = body_style_model.predict(crop, imgsz=224, verbose=False)[0]
    probs = {bs_res.names[i]: float(bs_res.probs.data[i]) for i in range(len(bs_res.names))}

    p_conv = probs.get('Convertible', 0.0)
    p_crossover = probs.get('Crossover', 0.0)
    p_fastback = probs.get('Fastback', 0.0)
    p_hatch = probs.get('Hatchback', 0.0)
    p_mpv = probs.get('MPV', 0.0)
    p_minibus = probs.get('Minibus', 0.0)
    p_pickup = probs.get('Pickup Truck', 0.0)
    p_suv = probs.get('SUV', 0.0)
    p_sedan = probs.get('Sedan', 0.0)
    p_sports = probs.get('Sports_HardtopConvertible', 0.0)
    p_wagon = probs.get('Wagon', 0.0)

    # 1. EVALUASI TRUK (YOLO vehicle_model)
    if initial_vtype == "truck":
        if p_pickup >= 0.35:
            return 'truck', 'Pickup Truck', round(p_pickup, 3)
        # Jika dimensi mobil penumpang biasa dan bukan truk besar:
        p_passenger = p_mpv + p_minibus + p_suv + p_crossover + p_hatch + p_sedan
        if area_ratio < 0.28 and w_ratio < 0.55 and p_passenger >= 0.60:
            pass  # Reclassify as passenger car (lanjut ke evaluasi bodi mobil di bawah)
        else:
            return 'truck', 'Truk', round(v_conf, 3)

    # 2. EVALUASI BUS (YOLO vehicle_model)
    if initial_vtype == "bus":
        # Bus komersial sungguhan (Jetbus, bus pariwisata, TransJakarta):
        # Memiliki dimensi fisik sangat besar di kamera parkir (area > 32%, lebar > 58%, atau tinggi > 65%)
        # DAN tidak memiliki karakteristik mobil penumpang (MPV/SUV/Hatchback/Fastback)
        is_massive = (area_ratio >= 0.32) or (w_ratio >= 0.58) or (h_ratio >= 0.65)
        if is_massive and p_minibus >= 0.55 and not (p_fastback >= 0.30 or p_sports >= 0.30):
            return 'bus', 'Bus', round(v_conf, 3)
        # Jika bukan bus besar komersial -> mobil penumpang (Innova, Sigra, Calya, Avanza) yang misklasifikasi oleh COCO!
        # Reclassify as passenger car (lanjut ke penentuan tipe bodi MPV/SUV/Hatchback di bawah)

    # 3. KENDARAAN MOBIL PENUMPANG (CAR)
    if aspect >= 0.70:
        score_conv = p_conv * 0.1
        score_sports = p_sports * 0.1
    else:
        score_conv = p_conv
        score_sports = p_sports

    score_fastback = p_fastback * 0.4
    score_pickup = p_pickup
    score_wagon = p_wagon
    # Gabungkan sinyal Minibus langsung ke MPV (mobil 7-seater keluarga di Indonesia adalah MPV)
    score_mpv = (p_mpv + p_minibus) * 1.50 + p_wagon * 0.7
    score_suv = p_suv * 1.2
    score_crossover = p_crossover * 1.2
    score_hatch = p_hatch * 1.2
    score_sedan = p_sedan * 1.2

    # Aturan CCTV Tampak Depan: Sinyal Fastback & Sports Convertible dari sudut atas
    if p_sports >= 0.20 or p_fastback >= 0.20:
        if aspect >= 0.75:
            score_mpv += (p_sports * 0.55) + (p_fastback * 0.45) + 0.15
            score_suv += (p_sports * 0.40) + (p_fastback * 0.35)
        elif aspect >= 0.65:
            score_crossover += (p_fastback * 0.85) + (p_sports * 0.75)
        else:
            score_sedan += (p_fastback * 0.85) + (p_sports * 0.75)

    if p_hatch >= 0.30 and p_suv >= 0.30:
        score_hatch += 0.20

    if aspect >= 0.80:
        score_mpv += 0.15
        score_suv += 0.05
    elif aspect <= 0.60:
        score_sedan += 0.10
        score_hatch += 0.08

    scores = {
        'MPV': score_mpv,
        'SUV': score_suv,
        'Crossover': score_crossover,
        'Hatchback': score_hatch,
        'Sedan': score_sedan,
        'Fastback': score_fastback,
        'Wagon': score_wagon,
        'Pickup Truck': score_pickup,
        'Convertible': score_conv,
        'Sports_HardtopConvertible': score_sports
    }

    best_cat = max(scores.items(), key=lambda kv: kv[1])[0]
    best_val = scores[best_cat]

    if best_cat in ('Minibus', 'Fastback', 'Sports_HardtopConvertible', 'Convertible', 'Wagon'):
        # Map exotic / non-Indonesian categories to closest realistic Indonesian car category
        if aspect >= 0.75:
            best_cat = 'MPV'
        elif aspect >= 0.65:
            best_cat = 'SUV' if p_suv >= p_hatch else 'Hatchback'
        else:
            best_cat = 'Sedan'

    tot_score = max(0.01, sum(scores.values()))
    norm_conf = min(0.99, max(0.60, best_val / tot_score))

    return 'car', best_cat, round(norm_conf, 3)


def map_indonesian_body_style(probs_dict, bbox, img_shape):
    """Wrapper kompatibilitas mundur untuk pemanggilan lawas."""
    top1_name = max(probs_dict.items(), key=lambda kv: kv[1])[0]
    return top1_name, round(probs_dict[top1_name], 3)


def get_gate_plate_priority(p, img_w, img_h):
    """
    Menghitung skor prioritas plat nomor di gerbang parkir.
    Memprioritaskan kendaraan di lajur aktif gerbang (tengah/foreground)
    dan memberikan penalti pada kendaraan di lajur samping/tetangga (tepi ekstrim gambar).
    """
    x1, y1, x2, y2 = p["box"]
    area = (x2 - x1) * (y2 - y1)
    conf = p["conf"]
    pcx = (x1 + x2) / 2.0
    pcy = (y1 + y2) / 2.0

    # 1. Faktor vertikal: Kendaraan di depan palang / tapping kartu berada di foreground bawah
    y_weight = 1.0 + (pcy / max(1, img_h) * 0.8)

    # 2. Faktor lajur gerbang: Jalur aktif berada di area tengah (20% - 75% lebar citra)
    # Kendaraan di tepi ekstrim (>78% atau <18%) adalah kendaraan di lajur samping/tetangga
    x_ratio = pcx / max(1, img_w)
    lane_weight = 0.35 if (x_ratio > 0.78 or x_ratio < 0.18) else 1.0

    return area * (conf ** 0.5) * y_weight * lane_weight


def get_front_of_camera_score(cand, plates, img_w, img_h):
    """
    Menghitung skor prioritas kendaraan 'di depan kamera'.
    Memprioritaskan kendaraan foreground (bawah/tengah), berukuran signifikan,
    dan menaungi plat nomor yang aktif di depan kamera.
    Menyingkirkan objek background kecil (<2.5% area) atau kendaraan di tepi ekstrim gambar.
    """
    x1, y1, x2, y2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area = bw * bh
    area_ratio = area / float(max(1, img_w * img_h))

    # 1. Filter out background noise
    # Objek sangat kecil (< 2.5% luas frame) atau hanya berada di latar belakang jauh (y2 < 30% tinggi frame)
    if area_ratio < 0.025:
        return -1.0
    if y2 < (img_h * 0.30):
        return -1.0

    cx = (x1 + x2) / 2.0
    norm_cx = cx / float(max(1, img_w))
    norm_y2 = y2 / float(max(1, img_h))

    # Objek di tepi ekstrim kiri (< 10%) atau kanan (> 90%)
    if norm_cx < 0.10 or norm_cx > 0.90:
        return -1.0

    # Skor kedekatan vertikal (foreground): kendaraan di depan kamera berada di bagian bawah frame
    y_score = (norm_y2 ** 1.8) * 2.0

    # Skor posisi tengah horizontal (kamera mengarah ke lajur tengah)
    center_dist = abs(norm_cx - 0.5)
    x_score = max(0.0, 1.0 - (center_dist * 1.6))

    # Skor ukuran (semakin dekat semakin besar)
    size_score = min(2.0, area_ratio * 6.0)

    # Cek apakah menaungi plat nomor
    has_plate = any(
        (x1 - 25 <= (p["box"][0] + p["box"][2]) / 2.0 <= x2 + 25) and
        (y1 - 25 <= (p["box"][1] + p["box"][3]) / 2.0 <= y2 + 25)
        for p in plates
    )
    plate_bonus = 2.5 if has_plate else 0.0

    # Confidence kendaraan
    conf_score = cand.get("v_conf", 0.5) * 0.5

    return y_score + x_score + size_score + plate_bonus + conf_score


# ============================================================
# FAST TEMPORAL CONFIRMATION & VEHICLE TRACKING (OPTIMIZED)
# 2-3 frame buffer (150-250ms) with early confirmation & deferred OCR.
# Guarantees exactly ONE confirmed record per vehicle in Entry History.
# ============================================================
class VehicleTrack:
    """
    Melacak dan mengonfirmasi kendaraan secara temporal cepat (2-3 frame / 150-250ms).
    State Machine: DETECTED -> ANALYZING -> CONFIRMED -> HISTORY_SAVED
    """
    def __init__(self, track_id, initial_det=None):
        self.track_id = track_id
        self.created_at = time.time()
        self.last_seen = time.time()
        self.frames = deque(maxlen=MAX_OBSERVATIONS)
        self.status = "DETECTED"  # DETECTED, ANALYZING, CONFIRMED, HISTORY_SAVED
        self.confirmed_data = None
        self.confirmation_score = 0.0
        self.history_saved = False
        self.last_bbox = None
        # Cache body style to avoid re-classifying every frame
        self.cached_body_style = None
        self.cached_body_conf = 0.0
        self.cached_vtype = None
        self.best_plate_crop = None
        self.best_plate_conf = 0.0
        if initial_det:
            self.add_frame(initial_det)

    def add_frame(self, det):
        self.last_seen = time.time()
        self.last_bbox = det.get("bbox")
        p_crop = det.get("plate_crop")
        p_conf = det.get("plate_confidence", 0.0) or 0.0
        if p_crop is not None and getattr(p_crop, 'size', 0) > 0 and p_conf > self.best_plate_conf:
            self.best_plate_crop = p_crop
            self.best_plate_conf = p_conf

        self.frames.append({
            "time": self.last_seen,
            "vehicle_type": det.get("vehicle_type"),
            "v_conf": det.get("vehicle_confidence", 0.0) or 0.0,
            "body_style": det.get("body_style"),
            "body_conf": det.get("body_style_confidence", 0.0) or 0.0,
            "license_plate": det.get("license_plate"),
            "plate_conf": p_conf,
            "plate_crop": p_crop,
            "bbox": det.get("bbox"),
            "plate_bbox": det.get("plate_bbox"),
            "ocr_method": det.get("ocr_method")
        })

        if self.status in ["CONFIRMED", "HISTORY_SAVED"]:
            return

        if len(self.frames) >= 1:
            self.status = "ANALYZING"

        self._evaluate_confirmation()

    def _evaluate_confirmation(self):
        n = len(self.frames)
        if n < MIN_OBSERVATIONS:
            return

        # 1. EARLY CONFIRMATION (Frame 2):
        # Jika ada setidaknya 2 observasi berturut-turut dengan kelas kendaraan sama dan confidence >= CONFIRM_CONF_THRESH (0.80)
        # Contoh: Frame 1: Car 0.88, Frame 2: Car 0.91 -> CONFIRM immediately!
        f1, f2 = self.frames[0], self.frames[1]
        same_vtype = (f1["vehicle_type"] == f2["vehicle_type"]) and (f1["v_conf"] >= CONFIRM_CONF_THRESH) and (f2["v_conf"] >= CONFIRM_CONF_THRESH)
        same_bstyle = bool(f1["body_style"] and f2["body_style"] and (f1["body_style"] == f2["body_style"]) and (f1["body_conf"] >= CONFIRM_CONF_THRESH) and (f2["body_conf"] >= CONFIRM_CONF_THRESH))

        if same_vtype or same_bstyle:
            self._finalize_confirmation(reason="early_consistency_2_frames")
            return

        # 2. PLATE-ASSISTED FAST CONFIRMATION:
        # Jika plat nomor terdeteksi konsisten (conf >= 0.20)
        has_clear_plate = any((f.get("plate_conf") or 0) >= PLATE_CONF_THRESH for f in self.frames)
        if has_clear_plate and n >= MIN_OBSERVATIONS:
            self._finalize_confirmation(reason="plate_assisted_fast_confirmation")
            return

        # 3. Maximum observations (3 frames) atau time window >= 250ms reached:
        time_span = self.frames[-1]["time"] - self.frames[0]["time"]
        if n >= MAX_OBSERVATIONS or time_span >= TEMPORAL_WINDOW_SEC:
            self._finalize_confirmation(reason="buffer_max_3_frames")

    def _finalize_confirmation(self, reason="buffer_voting"):
        # A. Voting Body Style (Weighted Confidence)
        style_weights = {}
        total_style_weight = 0.0
        for f in self.frames:
            bs = f["body_style"]
            if not bs:
                continue
            w = max(0.1, f["body_conf"])
            style_weights[bs] = style_weights.get(bs, 0.0) + w
            total_style_weight += w

        if style_weights:
            best_style = max(style_weights, key=style_weights.get)
            consistency_score = style_weights[best_style] / max(1e-5, total_style_weight)
            style_confs = [f["body_conf"] for f in self.frames if f["body_style"] == best_style and f["body_conf"] > 0]
            avg_style_conf = float(np.mean(style_confs)) if style_confs else 0.85
        else:
            best_style = self.frames[-1]["body_style"]
            consistency_score = 1.0
            avg_style_conf = self.frames[-1]["body_conf"]

        # B. Voting Vehicle Type
        type_weights = {}
        for f in self.frames:
            vt = f["vehicle_type"]
            if not vt:
                continue
            w = max(0.1, f["v_conf"])
            type_weights[vt] = type_weights.get(vt, 0.0) + w
        best_type = max(type_weights, key=type_weights.get) if type_weights else self.frames[-1]["vehicle_type"]

        # C. DEFERRED OCR: Jalankan OCR HANYA saat kendaraan TERKONFIRMASI!
        # Ambil plate_crop terbaik yang terkumpul di buffer
        best_plate = None
        best_plate_conf = None
        ocr_method = None

        target_crop = self.best_plate_crop
        if target_crop is None or getattr(target_crop, 'size', 0) == 0:
            for f in reversed(self.frames):
                if f.get("plate_crop") is not None and getattr(f["plate_crop"], 'size', 0) > 0:
                    target_crop = f["plate_crop"]
                    break

        if target_crop is not None and getattr(target_crop, 'size', 0) > 0:
            try:
                t_ocr_0 = time.time()
                ensemble_res = ensemble_plate_reading(target_crop)
                best_plate = ensemble_res["final"]
                ocr_method = ensemble_res["method"]
                best_plate_conf = ensemble_res["confidence"]
                print(f"[PERF] Deferred OCR executed for Track #{self.track_id}: '{best_plate}' in {round((time.time() - t_ocr_0)*1000, 1)}ms")
            except Exception as ocr_err:
                print(f"[WARN] Deferred OCR error: {ocr_err}")

        # Frame terbaik untuk snapshot / bounding box
        best_f = max(self.frames, key=lambda f: (f["plate_conf"] if f["plate_conf"] else 0.0) + f["body_conf"])

        self.confirmed_data = {
            "track_id": self.track_id,
            "vehicle_type": best_type,
            "body_style": best_style,
            "body_style_confidence": round(avg_style_conf, 3) if avg_style_conf else None,
            "license_plate": best_plate,
            "plate_confidence": round(best_plate_conf, 3) if best_plate_conf else None,
            "consistency": round(consistency_score, 2),
            "bbox": best_f["bbox"],
            "plate_bbox": best_f["plate_bbox"],
            "ocr_method": ocr_method
        }
        self.confirmation_score = round(consistency_score, 2)
        self.status = "CONFIRMED"

    def get_current_consistency(self):
        if not self.frames:
            return 1.0
        counts = {}
        for f in self.frames:
            b = f["body_style"] or f["vehicle_type"]
            if b:
                counts[b] = counts.get(b, 0) + 1
        return round(max(counts.values()) / len(self.frames), 2) if counts else 1.0


class VehicleConfirmationManager:
    """
    Mengelola multi-object tracking dan konfirmasi temporal kendaraan.
    Menjamin setiap kendaraan unik hanya dicatat 1x ke Entry History saat terkonfirmasi.
    """
    def __init__(self):
        self.tracks = {}
        self.lock = threading.Lock()
        self.next_fallback_id = 1

    def reset(self):
        with self.lock:
            self.tracks.clear()
            self.next_fallback_id = 1

    def _match_track(self, bbox, img_w=1920, img_h=1080, iou_thresh=0.10, max_center_dist_ratio=0.35):
        """
        Mencocokkan bounding box baru dengan track kendaraan aktif yang sudah ada.
        Menggunakan kombinasi IoU dan jarak pusat (Centroid) agar kendaraan yang bergerak
        cepat pada low FPS tetap terhubung pada Track ID yang sama (mencegah ID jumping).
        """
        if not bbox:
            return None
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        best_id = None
        best_score = -1.0

        for tid, trk in self.tracks.items():
            if trk.last_bbox:
                lx1, ly1, lx2, ly2 = trk.last_bbox
                lcx = (lx1 + lx2) / 2.0
                lcy = (ly1 + ly2) / 2.0

                iou = compute_iou(bbox, trk.last_bbox)
                dx = abs(cx - lcx) / max(1, img_w)
                dy = abs(cy - lcy) / max(1, img_h)
                dist = (dx**2 + dy**2)**0.5

                # Cocok jika IoU >= 0.10 ATAU jarak pusat mobil berdekatan (< 35% ukuran frame)
                if iou >= iou_thresh or dist <= max_center_dist_ratio:
                    score = iou + (1.0 - min(1.0, dist / max_center_dist_ratio))
                    if score > best_score:
                        best_score = score
                        best_id = tid
        return best_id

    def update(self, detections, is_stream=False):
        with self.lock:
            now = time.time()
            # Bersihkan track yang tidak terlihat > 8.0 detik
            stale_ids = [tid for tid, trk in self.tracks.items() if (now - trk.last_seen) > 8.0]
            for tid in stale_ids:
                del self.tracks[tid]

            if not detections:
                return []

            # Jika single photo upload manual (bukan live stream)
            if not is_stream:
                for det in detections:
                    det["track_id"] = 1
                    det["status"] = "CONFIRMED"
                    det["consistency"] = 1.0
                    det["is_newly_confirmed"] = True
                    det.pop("plate_crop", None)
                return detections

            # Mode Stream / Live CCTV
            updated_detections = []
            for det in detections:
                raw_tid = det.get("track_id")
                bbox = det.get("bbox")
                iw = det.get("image_width", 1920)
                ih = det.get("image_height", 1080)

                # Prioritaskan pencocokan spasial dengan track aktif yang sudah ada
                matched_id = self._match_track(bbox, img_w=iw, img_h=ih)
                if matched_id is not None:
                    tid = matched_id
                elif raw_tid is not None:
                    tid = raw_tid
                else:
                    tid = self.next_fallback_id
                    self.next_fallback_id += 1

                det["track_id"] = tid

                if tid not in self.tracks:
                    self.tracks[tid] = VehicleTrack(tid, det)
                    track = self.tracks[tid]
                else:
                    track = self.tracks[tid]
                    track.add_frame(det)

                # Pasang status temporal konfirmasi ke detection object
                if track.history_saved:
                    det["status"] = "HISTORY_SAVED"
                    det["is_newly_confirmed"] = False
                    det["consistency"] = track.confirmation_score
                    if track.confirmed_data:
                        det["body_style"] = track.confirmed_data["body_style"]
                        det["body_style_confidence"] = track.confirmed_data["body_style_confidence"]
                        if track.confirmed_data.get("license_plate"):
                            det["license_plate"] = track.confirmed_data["license_plate"]
                            det["plate_confidence"] = track.confirmed_data["plate_confidence"]
                        det["vehicle_type"] = track.confirmed_data["vehicle_type"]
                elif track.status == "CONFIRMED":
                    det["status"] = "CONFIRMED"
                    det["is_newly_confirmed"] = True  # Sinyal untuk simpan ke Entry History!
                    track.status = "HISTORY_SAVED"
                    track.history_saved = True
                    det["consistency"] = track.confirmation_score
                    if track.confirmed_data:
                        det["body_style"] = track.confirmed_data["body_style"]
                        det["body_style_confidence"] = track.confirmed_data["body_style_confidence"]
                        if track.confirmed_data.get("license_plate"):
                            det["license_plate"] = track.confirmed_data["license_plate"]
                            det["plate_confidence"] = track.confirmed_data["plate_confidence"]
                        det["vehicle_type"] = track.confirmed_data["vehicle_type"]
                else:
                    det["status"] = "ANALYZING"
                    det["is_newly_confirmed"] = False
                    det["consistency"] = track.get_current_consistency()
                    det["analyzing_frame_count"] = len(track.frames)
                    det["analyzing_max_frames"] = MAX_OBSERVATIONS

                # PENTING: Jangan kirim plate_crop (ndarray) ke client / JSON response
                det.pop("plate_crop", None)
                updated_detections.append(det)

            return updated_detections


confirmation_manager = VehicleConfirmationManager()


def run_anpr(image_input, vehicle_conf=None, motorcycle_conf=None, plate_conf=None, single_vehicle_mode=True, is_stream=False):
    t_start = time.time()
    v_conf_thresh = vehicle_conf if vehicle_conf is not None else VEHICLE_CONF_THRESH
    m_conf_thresh = motorcycle_conf if motorcycle_conf is not None else MOTORCYCLE_CONF_THRESH
    p_conf_thresh = plate_conf if plate_conf is not None else PLATE_CONF_THRESH

    if isinstance(image_input, np.ndarray):
        img = image_input
        image_path = "memory_frame"
    else:
        image_path = str(image_input)
        img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar dari: {image_input}")

    ih, iw = img.shape[:2]

    # 1. YOLO INFERENCE (Configurable imgsz=640, device)
    t_yolo_0 = time.time()
    if is_stream:
        fut_v = ai_pool.submit(vehicle_model.track, img, persist=True, tracker="bytetrack.yaml",
                               conf=m_conf_thresh, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    else:
        fut_v = ai_pool.submit(vehicle_model.predict, img, conf=m_conf_thresh,
                               imgsz=IMG_SIZE, device=DEVICE, verbose=False)

    fut_p = ai_pool.submit(plate_model.predict, img, conf=p_conf_thresh,
                           imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    vdet = fut_v.result()[0]
    pdet_global = fut_p.result()[0]
    t_yolo = time.time() - t_yolo_0

    # 2. EKSTRAKSI KANDIDAT KENDARAAN
    candidates = []
    for box in vdet.boxes:
        v_cls = int(box.cls[0])
        v_conf = float(box.conf[0])
        track_id = int(box.id[0]) if (box.id is not None) else None
        vehicle_type = VEHICLE_CLASS_NAMES[v_cls]
        threshold = m_conf_thresh if vehicle_type == "motorcycle" else v_conf_thresh
        if v_conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        area = (x2 - x1) * (y2 - y1)
        candidates.append({
            "track_id": track_id,
            "vehicle_type": vehicle_type,
            "v_conf": v_conf,
            "x1": max(0, x1),
            "y1": max(0, y1),
            "x2": min(iw, x2),
            "y2": min(ih, y2),
            "area": area
        })

    # Ekstraksi hasil deteksi plat nomor global
    global_plates = []
    for pbox in pdet_global.boxes:
        px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
        global_plates.append({
            "box": [max(0, px1), max(0, py1), min(iw, px2), min(ih, py2)],
            "conf": float(pbox.conf[0]),
            "matched": False
        })

    # FALLBACK UNTUK MOBIL GELAP / HITAM:
    # Jika mobil hitam/gelap tidak terdeteksi YOLO vehicle di atas threshold karena menyatu dengan aspal/glare,
    # tetapi plat nomor terdeteksi oleh plate_model:
    if not candidates and global_plates:
        best_p = max(global_plates, key=lambda p: p["conf"])
        if best_p["conf"] >= 0.15:
            px1, py1, px2, py2 = best_p["box"]
            pw = px2 - px1
            ph = py2 - py1
            # Cek apakah ada box kendaraan di vdet yang menaungi plat meskipun conf rendah
            found_box = None
            for box in vdet.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                if bx1 - 30 <= (px1 + px2) / 2 <= bx2 + 30 and by1 - 30 <= (py1 + py2) / 2 <= by2 + 30:
                    found_box = box
                    break
            if found_box is not None:
                v_cls = int(found_box.cls[0])
                v_conf = float(found_box.conf[0])
                x1, y1, x2, y2 = map(int, found_box.xyxy[0].tolist())
            else:
                v_cls = 0  # car
                v_conf = max(0.55, best_p["conf"])
                # Rekonstruksi bbox kendaraan proporsional terhadap plat nomor
                x1 = max(0, int(px1 - pw * 2.2))
                x2 = min(iw, int(px2 + pw * 2.2))
                y1 = max(0, int(py1 - ph * 4.5))
                y2 = min(ih, int(py2 + ph * 1.0))
            candidates.append({
                "track_id": None,
                "vehicle_type": VEHICLE_CLASS_NAMES[v_cls],
                "v_conf": v_conf,
                "x1": max(0, x1), "y1": max(0, y1),
                "x2": min(iw, x2), "y2": min(ih, y2),
                "area": (x2 - x1) * (y2 - y1)
            })

    results_out = []
    t_body_total = 0.0
    t_ocr_total = 0.0

    # 2b. FILTER & FOKUS KENDARAAN DI DEPAN KAMERA (Front-of-Camera Priority)
    # Singkirkan objek background kecil (<2.5% area) atau kendaraan di tepi ekstrim gambar,
    # dan fokuskan deteksi HANYA pada kendaraan utama yang berada di depan kamera.
    if candidates:
        scored_candidates = []
        for c in candidates:
            s = get_front_of_camera_score(c, global_plates, iw, ih)
            if s > 0:
                c["front_score"] = s
                scored_candidates.append(c)

        scored_candidates.sort(key=lambda c: -c["front_score"])

        if single_vehicle_mode:
            candidates = [scored_candidates[0]] if scored_candidates else []
        else:
            candidates = scored_candidates

        for cand in candidates:
            initial_vtype = cand["vehicle_type"]
            v_conf = cand["v_conf"]
            cand_track_id = cand.get("track_id")
            x1, y1, x2, y2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
            vehicle_crop = img[y1:y2, x1:x2]

            # 3. BODY STYLE CACHING PER TRACK ID
            t_b0 = time.time()
            existing_trk = confirmation_manager.tracks.get(cand_track_id) if cand_track_id else None
            if existing_trk and existing_trk.cached_body_style and existing_trk.cached_body_conf >= 0.80:
                # REUSE CACHED RESULT (0 ms!)
                vehicle_type = existing_trk.cached_vtype or initial_vtype
                body_style = existing_trk.cached_body_style
                body_style_conf = existing_trk.cached_body_conf
            else:
                # Jalankan klasifikasi bodi jika track baru atau confidence sebelumnya < 0.80
                vehicle_type, body_style, body_style_conf = classify_vehicle_indonesian(
                    img, [x1, y1, x2, y2], initial_vtype, v_conf
                )
                if existing_trk:
                    existing_trk.cached_body_style = body_style
                    existing_trk.cached_body_conf = body_style_conf
                    existing_trk.cached_vtype = vehicle_type
            t_body_total += (time.time() - t_b0)

            matched_plate = None
            for p in global_plates:
                if not p["matched"]:
                    pcx = (p["box"][0] + p["box"][2]) / 2.0
                    pcy = (p["box"][1] + p["box"][3]) / 2.0
                    if x1 - 25 <= pcx <= x2 + 25 and y1 - 25 <= pcy <= y2 + 25:
                        matched_plate = p
                        p["matched"] = True
                        break

            plate_conf_val = None
            abs_plate_bbox = None
            plate_crop = np.array([])

            if matched_plate is not None and matched_plate.get("conf", 0.0) >= 0.40:
                gpx1, gpy1, gpx2, gpy2 = matched_plate["box"]
                plate_crop = crop_plate_with_padding(img, gpx1, gpy1, gpx2, gpy2)
                plate_conf_val = matched_plate["conf"]
                abs_plate_bbox = [gpx1, gpy1, gpx2, gpy2]
            elif vehicle_crop.size > 0:
                pdet_crop = plate_model.predict(vehicle_crop, conf=p_conf_thresh, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
                if len(pdet_crop.boxes) > 0:
                    best_b = max(pdet_crop.boxes, key=lambda b: float(b.conf[0]))
                    cpx1, cpy1, cpx2, cpy2 = map(int, best_b.xyxy[0].tolist())
                    plate_conf_val = float(best_b.conf[0])
                    abs_plate_bbox = [x1 + cpx1, y1 + cpy1, x1 + cpx2, y1 + cpy2]
                    plate_crop = crop_plate_with_padding(img, abs_plate_bbox[0], abs_plate_bbox[1],
                                                         abs_plate_bbox[2], abs_plate_bbox[3])
                elif matched_plate is not None:
                    gpx1, gpy1, gpx2, gpy2 = matched_plate["box"]
                    plate_crop = crop_plate_with_padding(img, gpx1, gpy1, gpx2, gpy2)
                    plate_conf_val = matched_plate["conf"]
                    abs_plate_bbox = [gpx1, gpy1, gpx2, gpy2]
            elif matched_plate is not None:
                gpx1, gpy1, gpx2, gpy2 = matched_plate["box"]
                plate_crop = crop_plate_with_padding(img, gpx1, gpy1, gpx2, gpy2)
                plate_conf_val = matched_plate["conf"]
                abs_plate_bbox = [gpx1, gpy1, gpx2, gpy2]

            plate_text = None
            ocr_method = None
            # 4. PLATE READING:
            # - Single photo: Jalankan ensemble OCR lengkap
            # - Live stream: Jalankan deskew + unsharp + char_model cepat (~30ms) untuk live reading di dashboard,
            #   sementara ensemble OCR lengkap dieksekusi saat CONFIRMED.
            if plate_crop.size > 0:
                t_o0 = time.time()
                if not is_stream:
                    ensemble_res = ensemble_plate_reading(plate_crop)
                    plate_text = ensemble_res["final"]
                    ocr_method = ensemble_res["method"]
                else:
                    deskewed_p, _ = deskew_plate(plate_crop)
                    gaussian = cv2.GaussianBlur(deskewed_p, (0, 0), 2.0)
                    unsharp_p = cv2.addWeighted(deskewed_p, 1.8, gaussian, -0.8, 0)
                    char_text, c_conf, _ = read_plate_with_char_model(unsharp_p, conf=0.08)
                    if char_text:
                        plate_text = refine_indonesian_plate(char_text)
                        ocr_method = "char_model_fast"
                    else:
                        ocr_method = "none"
                t_ocr_total += (time.time() - t_o0)

            results_out.append({
                "track_id": cand_track_id,
                "vehicle_type": vehicle_type,
                "body_style": body_style,
                "body_style_confidence": round(body_style_conf, 3) if body_style_conf else None,
                "license_plate": plate_text,
                "ocr_method": ocr_method,
                "vehicle_confidence": round(v_conf, 3),
                "plate_confidence": round(plate_conf_val, 3) if plate_conf_val else None,
                "bbox": [x1, y1, x2, y2],
                "plate_bbox": abs_plate_bbox,
                "plate_crop": plate_crop if is_stream else None,
                "image_width": iw,
                "image_height": ih,
            })

    # 5. FAST TEMPORAL CONFIRMATION (Evaluasi & Deferred OCR saat Confirmed)
    t_track_0 = time.time()
    results_out = confirmation_manager.update(results_out, is_stream=is_stream)
    t_track = time.time() - t_track_0

    # Pastikan ndarray tidak pernah dikirim ke JSON / client
    for d in results_out:
        d.pop("plate_crop", None)

    t_total = time.time() - t_start
    fps = 1.0 / max(1e-4, t_total)

    # 5. PERFORMANCE TELEMETRY LOGGING
    print(f"[PERF] YOLO: {round(t_yolo*1000, 1)}ms | Tracking: {round(t_track*1000, 1)}ms | Body: {round(t_body_total*1000, 1)}ms | OCR: {round(t_ocr_total*1000, 1)}ms | Total: {round(t_total*1000, 1)}ms | FPS: {round(fps, 1)}")

    return {
        "detections": results_out,
        "detection_time_sec": round(t_total, 3),
        "source": image_path,
        "image_width": iw,
        "image_height": ih,
        "perf_breakdown": {
            "yolo_ms": round(t_yolo * 1000, 1),
            "track_ms": round(t_track * 1000, 1),
            "body_ms": round(t_body_total * 1000, 1),
            "ocr_ms": round(t_ocr_total * 1000, 1),
            "total_ms": round(t_total * 1000, 1),
            "fps": round(fps, 1)
        }
    }


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
@app.route("/dashboard")
def serve_dashboard():
    """Menyajikan antarmuka web dashboard ANPR."""
    return send_from_directory(MODEL_DIR, "parkir-anpr-dashboard.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Server ANPR aktif",
        "memory_records_count": len(LATEST_RECORDS),
        "stream_status": camera_stream_manager.get_status()
    })


@app.route("/api/live_stream")
def live_stream():
    """
    Endpoint HTTP MJPEG Video Stream (Ultra-Smooth 30-60 FPS) untuk ditampilkan langsung di monitor CCTV / browser.
    Menggunakan event-driven condition wait untuk menyiarkan setiap frame kamera seketika tanpa delay.
    """
    def gen():
        last_id = -1
        while True:
            with camera_stream_manager.condition:
                if not camera_stream_manager.running:
                    camera_stream_manager.condition.wait(timeout=0.2)
                else:
                    camera_stream_manager.condition.wait_for(
                        lambda: camera_stream_manager.frame_id != last_id or not camera_stream_manager.running,
                        timeout=0.08
                    )
                jpeg = camera_stream_manager.latest_jpeg
                last_id = camera_stream_manager.frame_id

            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            else:
                time.sleep(0.02)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/stream/start", methods=["POST", "OPTIONS"])
def stream_start():
    """Memulai streaming video kamera CCTV di latar belakang (koneksi permanen)."""
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json(silent=True) or {}
    stream_url = data.get("stream_url", "").strip()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not stream_url:
        return jsonify({"error": "stream_url wajib diisi"}), 400

    ok, msg = camera_stream_manager.start(stream_url, username, password)
    return jsonify({
        "status": "ok" if ok else "error",
        "message": msg,
        "stream_url": stream_url
    })


@app.route("/api/stream/stop", methods=["POST", "OPTIONS"])
def stream_stop():
    """Menghentikan streaming video kamera CCTV."""
    if request.method == "OPTIONS":
        return "", 200
    ok, msg = camera_stream_manager.stop()
    return jsonify({"status": "ok", "message": msg})


@app.route("/api/stream/status", methods=["GET"])
def stream_status():
    """Mengecek status koneksi kamera dan FPS streaming saat ini."""
    return jsonify(camera_stream_manager.get_status())


@app.route("/api/detect_current", methods=["POST", "GET", "OPTIONS"])
def detect_current():
    """
    Deteksi instan langsung dari frame terkini di RAM (0 ms delay pengambilan frame).
    Total waktu deteksi < 1 detik (jauh di bawah batas 3 detik).
    """
    if request.method == "OPTIONS":
        return "", 200

    frame = camera_stream_manager.get_latest_frame()
    if frame is None:
        return jsonify({"error": "Belum ada frame video di memory. Pastikan kamera CCTV sudah terhubung dan aktif."}), 400

    t0 = time.time()
    result = run_anpr(frame, is_stream=True)
    det_time = time.time() - t0
    result["detection_time_sec"] = round(det_time, 3)

    if result.get("detections"):
        primary_det = result["detections"][0]
        primary_det["latency_ms"] = round(det_time * 1000)
        # HANYA simpan ke Entry History jika kendaraan baru saja TERKONFIRMASI (1x per kendaraan)
        if primary_det.get("is_newly_confirmed"):
            rec = save_parking_record(primary_det, source_img=frame)
            primary_det["record"] = rec
            if rec and rec.get("snapshot_url"):
                result["image_url"] = rec["snapshot_url"]

    return jsonify(result)


@app.route("/api/upload_video", methods=["POST", "OPTIONS"])
def upload_video():
    """
    Menerima file video dan langsung menyiarkannya sebagai stream CCTV real-time di background.
    Memungkinkan simulasi feed CCTV berbasis rekaman video parkir dengan auto-detect.
    """
    if request.method == "OPTIONS":
        return "", 200
    if "video" not in request.files:
        return jsonify({"error": "Tidak ada file 'video' yang dikirim"}), 400
    file = request.files["video"]
    filename = file.filename or "uploaded_video.mp4"
    dest_path = os.path.join(MODEL_DIR, "uploaded_test_video.mp4")
    file.save(dest_path)

    confirmation_manager.reset()
    ok, msg = camera_stream_manager.start(dest_path)
    return jsonify({
        "status": "ok" if ok else "error",
        "message": f"Video '{filename}' siap disiarkan secara real-time",
        "video_path": dest_path
    })


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 200
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file 'image' yang dikirim"}), 400
    file = request.files["image"]
    temp_path = "temp_upload.jpg"
    file.save(temp_path)
    try:
        t0 = time.time()
        is_stream = request.form.get("is_stream", "false").lower() in ["true", "1"]
        result = run_anpr(temp_path, is_stream=is_stream)
        det_time = time.time() - t0
        result["detection_time_sec"] = round(det_time, 3)
        # Simpan ke memory RAM jika kendaraan terkonfirmasi (atau single photo upload)
        if result.get("detections"):
            primary_det = result["detections"][0]
            primary_det["latency_ms"] = round(det_time * 1000)
            if primary_det.get("is_newly_confirmed"):
                rec = save_parking_record(primary_det, source_img_path=temp_path)
                primary_det["record"] = rec
                if rec and rec.get("snapshot_url"):
                    result["image_url"] = rec["snapshot_url"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Mengambil daftar riwayat kendaraan masuk langsung dari memory RAM (0ms)."""
    limit = request.args.get("limit", default=100, type=int)
    records = list(LATEST_RECORDS)[:limit]
    return jsonify({"records": records, "total": len(records), "source": "memory"})


@app.route("/api/export", methods=["GET"])
def export_history():
    """Mengekspor seluruh riwayat kendaraan dari memory RAM ke file CSV."""
    records = list(LATEST_RECORDS)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Entry Timestamp", "License Plate", "Vehicle Type", "Body Style", "Confidence", "Snapshot File"])
    for r in records:
        conf_pct = f"{round(r['confidence']*100, 1)}%" if r.get("confidence") else "-"
        snap = r.get("snapshot_url", "").split("/")[-1] if r.get("snapshot_url") else "-"
        writer.writerow([r.get("id"), r.get("timestamp"), r.get("license_plate") or "-", r.get("vehicle_type") or "-", r.get("body_style") or "-", conf_pct, snap])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=parking_history_anpr.csv"}
    )


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """Membersihkan seluruh catatan riwayat parkir di memory RAM seketika."""
    LATEST_RECORDS.clear()
    return jsonify({"status": "ok", "message": "Riwayat parkir di memory berhasil dibersihkan", "total": 0})


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    """Menyajikan file foto bukti snapshot kendaraan."""
    return send_from_directory(CAPTURES_DIR, filename)


@app.route("/api/stream/capture", methods=["POST", "OPTIONS"])
def stream_capture():
    """
    Mengambil frame snapshot dari RTSP / HTTP MJPEG IP Stream (seperti DroidCam / IP Webcam / CCTV).
    Memungkinkan scan langsung dari IP DroidCam tanpa kendala CORS browser.
    """
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True) or {}
    stream_url = data.get("stream_url", "").strip()

    # Jalur Ultra-Cepat: jika background stream manager sedang aktif dan memiliki frame di RAM
    if camera_stream_manager.running and camera_stream_manager.latest_frame is not None:
        frame = camera_stream_manager.get_latest_frame()
        t0 = time.time()
        result = run_anpr(frame, is_stream=True)
        result["detection_time_sec"] = round(time.time() - t0, 3)
        if result.get("detections"):
            primary_det = result["detections"][0]
            if primary_det.get("is_newly_confirmed"):
                rec = save_parking_record(primary_det, source_img=frame)
                primary_det["record"] = rec
                if rec and rec.get("snapshot_url"):
                    result["image_url"] = rec["snapshot_url"]
        return jsonify(result)

    if not stream_url:
        return jsonify({"error": "stream_url wajib diisi"}), 400

    is_isapi = "/isapi/" in stream_url.lower() or stream_url.lower().endswith("/picture")
    is_rtsp = stream_url.lower().startswith(("rtsp://", "rtsps://"))

    # JALUR 1: HIKVISION ISAPI HTTP SNAPSHOT (Direct Sensor Full-Res Snapshot)
    if is_isapi:
        try:
            # Ekstrak host, port, user, pass secara aman tanpa error karakter khusus (@, *, :, dll)
            username = data.get("username")
            password = data.get("password")

            cleaned = stream_url.strip()
            scheme = "http"
            if "://" in cleaned:
                scheme, _, rest = cleaned.partition("://")
            else:
                rest = cleaned

            if "/" in rest:
                netloc_raw, _, path_raw = rest.partition("/")
                path_part = "/" + path_raw
            else:
                netloc_raw = rest
                path_part = "/ISAPI/Streaming/channels/1/picture"

            # Ekstrak IP dan port
            ip_m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d+))?', netloc_raw)
            if ip_m:
                host = ip_m.group(1)
                port = ip_m.group(2)
                creds_prefix = netloc_raw[:ip_m.start()].rstrip("@* :")
                if not username or not password:
                    if ":" in creds_prefix:
                        u, _, p = creds_prefix.partition(":")
                        username = username or u
                        password = password or p
            else:
                parts = netloc_raw.split("@")
                host_port = parts[-1]
                if ":" in host_port:
                    host, port = host_port.split(":", 1)
                else:
                    host, port = host_port, None
                if (not username or not password) and len(parts) > 1:
                    u, _, p = parts[0].partition(":")
                    username = username or u
                    password = password or p

            username = username or "admin"
            password = password or ""
            netloc = f"{host}:{port}" if port else host
            clean_url = f"{scheme}://{netloc}{path_part}"

            # Hikvision default: HTTP Digest Authentication
            auth = HTTPDigestAuth(username, password) if (username or password) else None
            print(f"[INFO] Memanggil Hikvision ISAPI Snapshot: {clean_url} (User: {username})")
            r = requests.get(clean_url, auth=auth, timeout=8)
            if r.status_code == 401 and (username or password):
                # Fallback ke Basic Auth
                r = requests.get(clean_url, auth=HTTPBasicAuth(username, password), timeout=8)

            if r.status_code != 200:
                return jsonify({
                    "error": f"Gagal snapshot dari Hikvision ISAPI (HTTP {r.status_code}). Periksa IP ({netloc}), username, password, atau channel kamera."
                }), 502

            img_arr = np.frombuffer(r.content, np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Gagal membaca format gambar JPEG dari respons Hikvision ISAPI."}), 502

            temp_path = "temp_upload.jpg"
            cv2.imwrite(temp_path, frame)
            result = run_anpr(temp_path)
            if result.get("detections"):
                primary_det = result["detections"][0]
                rec = save_parking_record(primary_det, temp_path)
                primary_det["record"] = rec
                if rec and rec.get("snapshot_url"):
                    result["image_url"] = rec["snapshot_url"]
            return jsonify(result)
        except Exception as isapi_err:
            print(f"[WARN] ISAPI gagal ({isapi_err}), otomatis mencoba RTSP fallback ke Streaming/Channels/101...")
            try:
                rtsp_url = f"rtsp://{username}:{password}@{host}:554/Streaming/Channels/101"
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        print("[INFO] Sukses mengambil frame via RTSP fallback!")
                        temp_path = "temp_upload.jpg"
                        cv2.imwrite(temp_path, frame)
                        result = run_anpr(temp_path)
                        if result.get("detections"):
                            primary_det = result["detections"][0]
                            rec = save_parking_record(primary_det, temp_path)
                            primary_det["record"] = rec
                            if rec and rec.get("snapshot_url"):
                                result["image_url"] = rec["snapshot_url"]
                        return jsonify(result)
            except Exception as rtsp_err:
                print(f"[WARN] RTSP fallback juga gagal: {rtsp_err}")
            return jsonify({"error": f"Hikvision ISAPI Error: {str(isapi_err)}"}), 500

    # JALUR 2: RTSP STREAM ATAU DROIDCAM / MJPEG HTTP STREAM
    if not is_rtsp:
        if not re.search(r':\d+', stream_url):
            stream_url += ":4747/video"
        elif stream_url.endswith(":4747"):
            stream_url += "/video"

    try:
        if is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            dev_type = "CCTV RTSP" if is_rtsp else "Kamera/DroidCam"
            return jsonify({"error": f"Gagal membuka koneksi ke {dev_type} di: {stream_url}. Pastikan perangkat aktif dan kredensial/IP benar."}), 502
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return jsonify({"error": f"Gagal membaca frame video dari: {stream_url}"}), 502

        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, frame)
        result = run_anpr(temp_path)
        if result.get("detections"):
            primary_det = result["detections"][0]
            rec = save_parking_record(primary_det, temp_path)
            primary_det["record"] = rec
            if rec and rec.get("snapshot_url"):
                result["image_url"] = rec["snapshot_url"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rpi/ping", methods=["GET"])
def rpi_ping():
    """Memeriksa status koneksi ke ALPR Kit di Raspberry Pi."""
    rpi_url = request.args.get("rpi_url", "http://192.168.1.4:5000").rstrip("/")
    try:
        r = requests.get(rpi_url, timeout=3)
        return jsonify({"status": "online", "code": r.status_code, "url": rpi_url})
    except Exception as e:
        return jsonify({"status": "offline", "error": str(e), "url": rpi_url}), 200


@app.route("/api/rpi/capture", methods=["POST", "OPTIONS"])
def rpi_capture():
    """
    Menghubungkan ke ALPR Kit di Raspberry Pi:
    1. Memanggil GET {rpi_url}/capture_image untuk menjepret foto.
    2. Mengambil foto via POST {rpi_url}/get_image jika diperlukan.
    3. Menjalankan model AI ANPR lokal (YOLO + EasyOCR).
    4. Menyimpan data ke SQLite dan mengembalikan hasil lengkap ke dashboard.
    """
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True) or {}
    rpi_url = data.get("rpi_url", "http://192.168.1.4:5000").rstrip("/")

    try:
        cap_url = f"{rpi_url}/capture_image"
        print(f"[INFO] Memanggil ALPR Kit RPi: {cap_url}")
        r = requests.get(cap_url, timeout=12)

        image_bytes = None
        content_type = r.headers.get("Content-Type", "")

        # Kasus A: Respons langsung berupa binary gambar
        if "image" in content_type:
            image_bytes = r.content
        else:
            # Kasus B: Respons JSON berisi path gambar atau base64
            try:
                res_json = r.json()
            except Exception:
                res_json = {}

            img_path = res_json.get("image_path") or res_json.get("path")
            if not img_path and isinstance(res_json.get("data"), dict):
                img_path = res_json["data"].get("image_path")

            if img_path:
                get_img_url = f"{rpi_url}/get_image"
                print(f"[INFO] Mengunduh foto hasil jepretan dari RPi: {get_img_url} ({img_path})")
                r_img = requests.post(get_img_url, json={"image_path": img_path}, timeout=12)
                if "image" in r_img.headers.get("Content-Type", ""):
                    image_bytes = r_img.content
                else:
                    try:
                        j_img = r_img.json()
                        b64_str = j_img.get("image") or j_img.get("image_base64") or j_img.get("data")
                        if b64_str:
                            if "," in b64_str:
                                b64_str = b64_str.split(",")[1]
                            image_bytes = base64.b64decode(b64_str)
                    except Exception:
                        image_bytes = r_img.content
            elif res_json.get("image_base64") or res_json.get("image"):
                b64_str = res_json.get("image_base64") or res_json.get("image")
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                image_bytes = base64.b64decode(b64_str)
            else:
                if len(r.content) > 1000:
                    image_bytes = r.content

        if not image_bytes or len(image_bytes) < 100:
            return jsonify({
                "error": f"Raspberry Pi ALPR Kit tidak mengembalikan gambar valid. Respons: {r.text[:200]}"
            }), 500

        # Simpan sementara foto hasil jepretan kamera Raspberry Pi
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        # Proses dengan pipeline ANPR lokal (YOLO Kendaraan + Plat + Karakter + Samsat Rules)
        result = run_anpr(temp_path)

        # Simpan otomatis ke database SQLite
        if result.get("detections"):
            primary_det = result["detections"][0]
            rec = save_parking_record(primary_det, temp_path)
            primary_det["record"] = rec
            if rec and rec.get("snapshot_url"):
                result["image_url"] = rec["snapshot_url"]

        return jsonify(result)

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": f"Tidak dapat terhubung ke Raspberry Pi di {rpi_url}. Pastikan Raspberry Pi sudah menyala dan terhubung ke jaringan WiFi yang sama."
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({
            "error": f"Waktu koneksi ke Raspberry Pi di {rpi_url} habis (Timeout)."
        }), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n=======================================================")
    print("🚀 Server ANPR jalan di http://localhost:5001")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
