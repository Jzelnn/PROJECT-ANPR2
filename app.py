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
from collections import deque, Counter
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import json
from urllib.parse import urlparse
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_sock import Sock

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
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")  # lewati cek konektivitas model hoster (PaddleOCR 3.x)
import paddleocr
from paddleocr import PaddleOCR

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
IMG_SIZE = int(os.environ.get("ANPR_IMG_SIZE", 512))           # 640 for reliable small plate detection
VEHICLE_CONF_THRESH = float(os.environ.get("ANPR_VEHICLE_CONF", 0.20))
MOTORCYCLE_CONF_THRESH = float(os.environ.get("ANPR_MOTOR_CONF", 0.08))
PLATE_CONF_THRESH = float(os.environ.get("ANPR_PLATE_CONF", 0.20))
IOU_THRESH = float(os.environ.get("ANPR_IOU_THRESH", 0.35))
FRAME_SKIP = int(os.environ.get("ANPR_FRAME_SKIP", 1))
PLATE_DETECT_REFRESH_FRAMES = int(os.environ.get("ANPR_PLATE_REFRESH_FRAMES", 4))         # Process 1 of every N frames (1 = all, 2 = half)
DEVICE = os.environ.get("ANPR_DEVICE", "cuda" if HAS_CUDA else "cpu")
USE_FP16 = False  # Keep false on CPU to prevent warnings

# Fast Temporal Confirmation Configuration
MIN_OBSERVATIONS = 3
MAX_OBSERVATIONS = 5
TEMPORAL_WINDOW_SEC = 0.50
CONSISTENCY_THRESH = 0.70
CONFIRM_CONF_THRESH = 0.80

# PaddleOCR & OCR Fusion Configuration
PADDLE_DEVICE = os.environ.get("ANPR_PADDLE_DEVICE", "cpu")            # "cpu" atau "gpu:0"
PADDLE_DET_MODEL = os.environ.get("ANPR_PADDLE_DET_MODEL") or None     # opsional (PaddleOCR 3.x)
PADDLE_REC_MODEL = os.environ.get("ANPR_PADDLE_REC_MODEL") or None     # opsional (PaddleOCR 3.x)
PADDLE_MIN_SHARPNESS = float(os.environ.get("ANPR_PADDLE_MIN_SHARPNESS", 4.0))  # var(Laplacian) minimum, BELUM dituning
PADDLE_PRIMARY_MARGIN = float(os.environ.get("ANPR_PADDLE_MARGIN", 0.10))
LOST_OCR_GRACE_SEC = float(os.environ.get("ANPR_LOST_OCR_GRACE", 1.5))  # tunggu job OCR berjalan saat LOST_INTEREST
DEBUG_PLATE_CROPS = os.environ.get("ANPR_DEBUG_CROPS", "0").lower() in ("1", "true", "yes")
DEBUG_CROPS_DIR = os.path.join(MODEL_DIR, "debug_plates")
# Pembacaan yang BUKAN hasil OCR baru tidak boleh dihitung sebagai observasi kandidat (mencegah self-reinforcement)
NON_EVIDENCE_OCR_METHODS = {"char_model_preview", "temporal_best_candidate", "locked_confirmed"}

# ============================================================
# DETECTION / INTEREST AREA CONFIGURATION (ROI)
# Kendaraan hanya menjadi target aktif ALPR setelah memasuki area ini.
# Bounding box normalisasi [x_min, y_min, x_max, y_max] (0.0 s/d 1.0).
# ============================================================
DEFAULT_INTEREST_AREA = {
    "x_min": float(os.environ.get("ANPR_ROI_XMIN", 0.12)),
    "y_min": float(os.environ.get("ANPR_ROI_YMIN", 0.20)),
    "x_max": float(os.environ.get("ANPR_ROI_XMAX", 0.88)),
    "y_max": float(os.environ.get("ANPR_ROI_YMAX", 0.95))
}
INTEREST_AREA = dict(DEFAULT_INTEREST_AREA)

# Lost Interest parameters
LOST_INTEREST_TIMEOUT_SEC = float(os.environ.get("ANPR_LOST_TIMEOUT", 2.0))
LOST_INTEREST_OUTSIDE_FRAMES = int(os.environ.get("ANPR_LOST_FRAMES", 4))


def is_vehicle_inside_roi(x1, y1, x2, y2, img_w, img_h, roi=None):
    """
    Menentukan apakah kendaraan berada di dalam Detection / Interest Area (ROI).
    Menggunakan titik tengah (cx, cy), titik bumper bawah (cx, y2), dan rasio overlap.
    """
    if roi is None:
        roi = INTEREST_AREA

    rx1 = roi["x_min"] * img_w
    ry1 = roi["y_min"] * img_h
    rx2 = roi["x_max"] * img_w
    ry2 = roi["y_max"] * img_h

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bx = cx
    by = float(y2)

    center_in = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
    bumper_in = (rx1 <= bx <= rx2) and (ry1 <= by <= ry2)
    if center_in or bumper_in:
        return True, 1.0

    ix1 = max(float(x1), rx1)
    iy1 = max(float(y1), ry1)
    ix2 = min(float(x2), rx2)
    iy2 = min(float(y2), ry2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    veh_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    overlap_ratio = inter_area / veh_area

    is_inside = overlap_ratio >= 0.35
    return is_inside, overlap_ratio


# ============================================================
# WEBSOCKET BROADCASTER FOR REAL-TIME ANPR TELEMETRY
# Mengirimkan pembaruan deteksi real-time & OCR ke seluruh frontend client yang terhubung.
# ============================================================
class WebSocketBroadcaster:
    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()

    def register(self, ws):
        with self.lock:
            self.clients.add(ws)
            print(f"[WS] Client connected. Total active clients: {len(self.clients)}")

    def unregister(self, ws):
        with self.lock:
            self.clients.discard(ws)
            print(f"[WS] Client disconnected. Total active clients: {len(self.clients)}")

    def broadcast(self, message_dict):
        with self.lock:
            if not self.clients:
                return
            msg_str = json.dumps(message_dict)
            dead = []
            for ws in list(self.clients):
                try:
                    ws.send(msg_str)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


ws_broadcaster = WebSocketBroadcaster()

# ============================================================
# PURE IN-MEMORY STORAGE (RAM)
# Seluruh riwayat kendaraan disimpan langsung di memori RAM (0 ms, bebas lag disk)
# ============================================================
LATEST_RECORDS = deque(maxlen=300)
LAST_RECORDED_PLATES = {}  # {safe_plate: {"time": float, "rec": dict}}
GATE_COOLDOWN_SEC = 15.0


def save_parking_record(det, source_img=None, source_img_path=None):
    """
    Menyimpan hasil deteksi yang TERKONFIRMASI langsung ke memory RAM (0ms).
    Persyaratan ketat Entry History:
    - Plat harus valid (format nomor polisi Indonesia atau militer)
    - Confidence plat minimal 80% (>= 0.80)
    - Anti-Passback Gate Cooldown (15 detik) untuk mencegah duplikasi catatan.
    """
    global LATEST_RECORDS, LAST_RECORDED_PLATES

    plate_text = det.get("license_plate")
    if not plate_text or plate_text in ["TIDAK_TERBACA", "UNKNOWN", ""]:
        return None

    if not isinstance(plate_text, str):
        plate_text = str(plate_text)
    safe_plate = re.sub(r'[^A-Za-z0-9]', '', plate_text)
    if len(safe_plate) < 4 and '-' not in plate_text:
        return None

    plate_conf = det.get("plate_confidence", 0.0) or 0.0
    is_confirmed = (det.get("status") in ["CONFIRMED", "HISTORY_SAVED"]) or det.get("is_newly_confirmed", False)
    min_thresh = 0.65 if is_confirmed else 0.80
    if plate_conf < min_thresh:
        return None

    now = datetime.datetime.now()
    now_epoch = time.time()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_ts = now.strftime("%Y%m%d_%H%M%S_%f")[:19]

    track_id = det.get("track_id")

    # Anti-Passback Gate Cooldown: Jika plat yang sama baru tercatat < 15 detik lalu, gunakan record yang ada
    if safe_plate in LAST_RECORDED_PLATES:
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

    rec = {
        "id": int(now_epoch * 1000) % 1000000,
        "track_id": track_id,
        "timestamp": timestamp_str,
        "license_plate": plate_text,
        "vehicle_type": det.get("vehicle_type"),
        "body_style": det.get("body_style"),
        "confidence": round(plate_conf, 3),
        "consistency": det.get("consistency", 1.0),
        "status": "CONFIRMED",
        "latency_ms": det.get("latency_ms"),
        "snapshot_url": f"/captures/{snapshot_filename}" if snapshot_filename else None
    }

    # Simpan langsung ke RAM (0 milidetik, tanpa overhead disk I/O)
    LATEST_RECORDS.appendleft(rec)
    LAST_RECORDED_PLATES[safe_plate] = {"time": now_epoch, "rec": rec}
    
    # Broadcast langsung ke WebSocket clients seketika
    ws_broadcaster.broadcast({
        "type": "entry_confirmed",
        "record": rec
    })

    conf_time = det.get("confirmation_time")
    dt_conf_to_hist = ((now_epoch - conf_time) * 1000.0) if conf_time else 0.0
    print(f"""[ENTRY HISTORY]
track={track_id}
plate=\"{safe_plate}\"
timestamp={now_epoch:.3f}
time_from_confirmation_to_history_ms={dt_conf_to_hist:.1f}
action=INSERTED""")
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
        self.latest_frame_time = None  # Timestamp when latest frame was captured
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
            if 'stream_inference_worker' in globals():
                stream_inference_worker.start()
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
        self.latest_frame_time = None
        if 'stream_inference_worker' in globals():
            stream_inference_worker.stop()
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
                    self.latest_frame_time = time.time()
                    self.frame_id += 1
                    self.condition.notify_all()

        cap.release()
        with self.lock:
            self.status = "disconnected"
        print("[STREAM] Koneksi video telah ditutup dengan aman.")


camera_stream_manager = CameraStreamManager()

class PaddleEngine:
    """
    Wrapper PaddleOCR tunggal. Diinisialisasi SEKALI saat startup (bukan per frame / per request).
    Mendukung PaddleOCR 3.x (.predict) dan 2.x (.ocr).
    Panggilan diserialkan dengan lock karena predictor Paddle tidak dijamin thread-safe
    (ocr_executor memakai 2 worker).
    run() mengembalikan list of {"text", "conf", "box"}; box = [[x, y] * 4] atau None.
    """
    def __init__(self, device="cpu", det_model=None, rec_model=None):
        self.lock = threading.Lock()
        self.version = getattr(paddleocr, "__version__", "unknown")
        self.api = "v3" if hasattr(PaddleOCR, "predict") else "v2"
        t0 = time.time()
        if self.api == "v3":
            kw = dict(
                lang="en",
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            if det_model:
                kw["text_detection_model_name"] = det_model
            if rec_model:
                kw["text_recognition_model_name"] = rec_model
            self.engine = PaddleOCR(**kw)
        else:
            self.engine = PaddleOCR(lang="en", use_angle_cls=False, use_gpu=(device != "cpu"), show_log=False)
        self.init_ms = (time.time() - t0) * 1000.0

        # Warm-up agar inferensi pertama (yang biasanya lambat) tidak terjadi saat kendaraan pertama lewat
        t1 = time.time()
        try:
            self.run(np.full((64, 224, 3), 255, dtype=np.uint8))
        except Exception as e:
            print(f"[PADDLE WARN] warm-up gagal: {e}")
        self.warmup_ms = (time.time() - t1) * 1000.0
        print(f"[PADDLE] siap | paddleocr={self.version} api={self.api} device={device} "
              f"init={self.init_ms:.0f}ms warmup={self.warmup_ms:.0f}ms")

    @staticmethod
    def _to_box(poly):
        try:
            arr = np.asarray(poly, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr.tolist()
            if arr.ndim == 1 and arr.size == 4:
                x1, y1, x2, y2 = arr.tolist()
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        except Exception:
            pass
        return None

    @staticmethod
    def _as_list(x):
        return [] if x is None else list(x)

    def _parse(self, out):
        items = []
        if self.api == "v3":
            for res in (out or []):
                d = res if isinstance(res, dict) else (getattr(res, "json", None) or {})
                if isinstance(d.get("res"), dict):
                    d = d["res"]
                texts = self._as_list(d.get("rec_texts"))
                scores = self._as_list(d.get("rec_scores"))
                polys = d.get("rec_polys")
                if polys is None:
                    polys = d.get("dt_polys")
                polys = self._as_list(polys)
                for i, t in enumerate(texts):
                    sc = float(scores[i]) if i < len(scores) else 0.0
                    box = self._to_box(polys[i]) if i < len(polys) else None
                    items.append({"text": str(t), "conf": sc, "box": box})
        else:
            page = out[0] if out else None
            for line in (page or []):
                try:
                    box, (t, sc) = line
                    items.append({"text": str(t), "conf": float(sc), "box": self._to_box(box)})
                except Exception:
                    continue
        return items

    def run(self, img_bgr):
        with self.lock:
            if self.api == "v3":
                out = self.engine.predict(img_bgr)
            else:
                out = self.engine.ocr(img_bgr, cls=False)
        return self._parse(out)


paddle_engine = None


print("[INFO] Memuat model AI...")
# 1. Model Deteksi Kendaraan Indonesia (dilatih khusus untuk kendaraan jalanan Indonesia)
indo_vmodel_path = os.path.join(MODEL_DIR, "vehicle_model_indo.pt")
if not os.path.exists(indo_vmodel_path):
    indo_vmodel_path = os.path.join(MODEL_DIR, "best (1).pt")

if os.path.exists(indo_vmodel_path):
    vehicle_model = YOLO(indo_vmodel_path)
    print(f"[INFO] Model Kendaraan Indonesia aktif: {os.path.basename(indo_vmodel_path)} ({len(vehicle_model.names)} kelas)")
else:
    vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_model.pt"))
    print(f"[INFO] Model Kendaraan Standar aktif: vehicle_model.pt ({len(vehicle_model.names)} kelas)")

# 2. Model Deteksi Plat Nomor & Sub-Tipe Bodi (PRKING-ANPR-1 Best Checkpoint)
best_plate_path = os.path.join(MODEL_DIR, "plate_detector_best.pt")
if not os.path.exists(best_plate_path):
    best_plate_path = os.path.join(MODEL_DIR, "best.pt")

if os.path.exists(best_plate_path):
    plate_detector = YOLO(best_plate_path)
    print(f"[INFO] Model Plat Nomor & Sub-Tipe Bodi Best aktif: {os.path.basename(best_plate_path)} ({len(plate_detector.names)} kelas)")
else:
    plate_detector = YOLO(os.path.join(MODEL_DIR, "plate_model.pt"))
    print(f"[INFO] Model Plat Nomor Standar aktif: plate_model.pt")

plate_model_legacy = YOLO(os.path.join(MODEL_DIR, "plate_model.pt"))
body_style_model = YOLO(os.path.join(MODEL_DIR, "body_style_model.pt"))
char_model = YOLO(os.path.join(MODEL_DIR, "char_model.pt"))
paddle_engine = PaddleEngine(device=PADDLE_DEVICE, det_model=PADDLE_DET_MODEL, rec_model=PADDLE_REC_MODEL)
ai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR_YOLO")
ocr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR_ASYNC_OCR")
ocr_job_counter = 0
ocr_job_lock = threading.Lock()

def get_next_ocr_job_id():
    global ocr_job_counter
    with ocr_job_lock:
        ocr_job_counter += 1
        return ocr_job_counter

print("[INFO] Semua model AI, PaddleOCR & Thread Pools siap digunakan!")


def map_vehicle_class_name(cls_id, model=vehicle_model):
    """Memetakan nama kelas deteksi YOLO kendaraan ke format standar: car, motorcycle, bus, truck."""
    raw_name = model.names.get(int(cls_id), "car").lower()
    if raw_name in ["mobil", "car"]:
        return "car"
    if raw_name in ["motor", "motorcycle"]:
        return "motorcycle"
    if raw_name in ["bus"]:
        return "bus"
    if raw_name in ["truck", "pickup"]:
        return "truck"
    return "car"


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


SAMSAT_PREFIXES = {
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
    'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN',
    'BP', 'DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DK', 'DM', 'DN', 'DP', 'DR', 'DT',
    'DW', 'EA', 'EB', 'ED', 'KB', 'KH', 'KT', 'KU'
}


def is_valid_indonesian_plate_structure(plate_str):
    """
    Memvalidasi apakah string OCR memenuhi struktur plat nomor Indonesia atau format dinas/militer.
    Returns: (is_valid: bool, plate_type: str or None)
    """
    if not plate_str or plate_str in ["TIDAK_TERBACA", "UNKNOWN", ""]:
        return False, None
    plate_str = str(plate_str).strip()
    # Format dinas / militer: e.g. "523-07", "1234-01"
    if '-' in plate_str:
        parts = plate_str.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return True, "military"
    clean = re.sub(r'[^A-Z0-9]', '', plate_str.upper())
    # Format sipil Indonesia: 1-2 huruf kode wilayah, 1-4 angka nomor polisi, 1-3 huruf seri akhir
    m = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', clean)
    if m:
        prefix, digits, suffix = m.groups()
        if prefix in SAMSAT_PREFIXES:
            return True, "standard"
        return True, "standard_general"
    return False, None


def is_valid_plate_box(box, img_w, img_h):
    """Memvalidasi geometri bounding box plat untuk menyingkirkan deteksi palsu (grille, bumper, garis aspal)."""
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / float(h)
    w_ratio = w / float(img_w)
    h_ratio = h / float(img_h)
    area_ratio = (w * h) / float(img_w * img_h)
    # Plat nomor Indonesia: rasio aspek ~ 1.15 - 6.2 (mendukung plat kotak TNI/Polri 1.2 - 1.5 & motor), lebar <= 45% frame, luas <= 10% frame
    if aspect < 1.15 or aspect > 6.2:
        return False
    if w_ratio > 0.45 or h_ratio > 0.28:
        return False
    if area_ratio > 0.10:
        return False
    return True


def compute_crop_quality(crop, p_conf=0.5):
    """
    Menghitung skor kualitas gambar plat nomor pada kendaraan bergerak.
    Memprioritaskan frame yang tajam (bebas motion blur) dan beresolusi cukup saat mobil mendekat.
    """
    if crop is None or getattr(crop, 'size', 0) == 0:
        return 0.0
    h, w = crop.shape[:2]
    if h < 14 or w < 28:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    res_factor = min(3.0, (h * w) / (32.0 * 95.0))
    aspect = w / float(max(1, h))
    aspect_factor = 1.0 if 2.0 <= aspect <= 5.0 else 0.6
    quality = res_factor * (sharpness ** 0.5) * (max(0.2, p_conf) ** 0.5) * aspect_factor
    return quality


def enhance_moving_plate_crop(crop):
    """
    Peningkatan ketajaman adaptif (Unsharp Masking) untuk mengurangi motion blur kendaraan yang sedang berjalan.
    Membuat kontur karakter angka dan huruf plat menjadi tegas dan kontras (<2ms CPU).
    """
    if crop is None or getattr(crop, 'size', 0) == 0:
        return crop
    gaussian = cv2.GaussianBlur(crop, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(crop, 1.5, gaussian, -0.5, 0)
    return sharpened


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

    # Baseline varians pada rotasi 0 derajat (abaikan 15% margin tepi agar terhindar dari artefak rotasi batas)
    margin_y = int(sh * 0.15)
    margin_x = int(sw * 0.10)
    sob0 = cv2.Sobel(gray[margin_y:sh-margin_y, margin_x:sw-margin_x], cv2.CV_64F, 0, 1, ksize=3)
    var0 = np.var(np.sum(np.abs(sob0), axis=1))

    best_var = var0
    best_ang = 0.0
    for a in np.arange(-8, 9, 1.0):
        if a == 0:
            continue
        M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), float(a), 1.0)
        rot = cv2.warpAffine(gray, M, (sw, sh), borderMode=cv2.BORDER_REPLICATE)
        rot_inner = rot[margin_y:sh-margin_y, margin_x:sw-margin_x]
        sob = cv2.Sobel(rot_inner, cv2.CV_64F, 0, 1, ksize=3)
        var = np.var(np.sum(np.abs(sob), axis=1))
        # Butuh peningkatan minimal 30% pada area interior plat agar tidak memiringkan plat normal
        if var > best_var * 1.30:
            best_var = var
            best_ang = float(a)

    if abs(best_ang) >= 3.0:
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
        # Disambiguasi 1 vs 2/7: karakter bergaris vertikal sangat sempit (stroke width < 0.42 * height)
        aspect_ratio = bw / max(1.0, bh)
        if cname in ['2', '7'] and aspect_ratio < 0.42:
            cname = '1'
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


# ============================================================
# PEMBACAAN PLAT NOMOR
# Dua sinyal independen:
#   1. PaddleOCR            -> engine OCR utama (paddle_text, paddle_confidence)
#   2. YOLO Character Model -> sinyal karakter-level (character_model_text, character_model_confidence)
# Confidence kedua sinyal TIDAK pernah dinaikkan secara artifisial; hasil akhir membawa
# confidence asli dari sumber yang dipilih. Bonus kesepakatan hanya masuk ke candidate_score (peringkat).
# ============================================================
def _alnum(text):
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())


def _save_debug_plate_crop(img, tag, suffix=""):
    """Simpan crop yang BENAR-BENAR dikirim ke PaddleOCR (aktif jika ANPR_DEBUG_CROPS=1)."""
    if not DEBUG_PLATE_CROPS or img is None or getattr(img, "size", 0) == 0:
        return
    try:
        os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%H%M%S_%f")[:10]
        cv2.imwrite(os.path.join(DEBUG_CROPS_DIR, f"{stamp}_{tag or 'plate'}{suffix}.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    except Exception:
        pass


def _prepare_paddle_input(crop):
    """Resize ringan ke tinggi yang nyaman bagi detektor teks Paddle + border kecil (tanpa filter agresif)."""
    h, w = crop.shape[:2]
    if h < 80:
        scale, interp = 96.0 / h, cv2.INTER_CUBIC
    elif h > 160:
        scale, interp = 128.0 / h, cv2.INTER_AREA
    else:
        scale, interp = 1.0, None
    if scale != 1.0:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)
    return cv2.copyMakeBorder(crop, 8, 8, 12, 12, cv2.BORDER_REPLICATE)


def _paddle_pass(img, label):
    """Satu pass PaddleOCR -> list segmen teks. Hasil mentah selalu dicatat ke log."""
    t0 = time.time()
    items = paddle_engine.run(img)
    ms = (time.time() - t0) * 1000.0
    H = max(1, img.shape[0])
    lines = []
    for it in items:
        raw = it["text"]
        conf = float(it["conf"])
        clean = re.sub(r'[^A-Z0-9\-]', '', str(raw).upper().replace('\u2013', '-').replace('\u2014', '-')).strip('-')
        box = it.get("box")
        if box:
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            cy, x0 = (sum(ys) / len(ys)) / H, min(xs)
        else:
            cy, x0 = None, None
        lines.append({"raw": raw, "clean": clean, "conf": conf, "cy": cy, "x0": x0})
        print(f"[PADDLE] pass={label} raw='{raw}' conf={conf:.3f} normalized='{clean}' "
              f"cy={'-' if cy is None else round(cy, 2)}")
    return lines, ms


def _merge_main_plate_line(lines):
    """
    Gabungkan segmen pada baris ATAS plat (nomor polisi). Baris bawah (bulan/tahun pajak) diabaikan.
    Detektor Paddle kadang memecah 'B 1591 BPC' menjadi beberapa segmen; segmen digabung kiri->kanan.
    Confidence gabungan = rata-rata tertimbang jumlah karakter (tanpa boost).
    """
    cand = [l for l in lines if _alnum(l["clean"]) and l["conf"] >= 0.10]
    if not cand:
        return "", 0.0
    boxed = [l for l in cand if l["cy"] is not None]
    if not boxed:
        best = max(cand, key=lambda l: l["conf"])
        return best["clean"], best["conf"]
    top = sorted([l for l in boxed if l["cy"] < 0.70], key=lambda l: l["cy"])
    if not top:
        return "", 0.0
    rows = []
    for l in top:
        if rows and abs(l["cy"] - (sum(r["cy"] for r in rows[-1]) / len(rows[-1]))) <= 0.15:
            rows[-1].append(l)
        else:
            rows.append([l])
    row = max(rows, key=lambda r: (sum(len(_alnum(x["clean"])) for x in r), sum(x["conf"] for x in r) / len(r)))
    row = sorted(row, key=lambda l: l["x0"] if l["x0"] is not None else 0.0)
    parts = [l["clean"] for l in row]
    if len(parts) == 2 and re.fullmatch(r'\d{3,4}', _alnum(parts[0])) and re.fullmatch(r'\d{2}', _alnum(parts[1])):
        text = f"{_alnum(parts[0])}-{_alnum(parts[1])}"   # format dinas/militer, misal 523-07
    else:
        text = "".join(parts)
    total = sum(len(_alnum(l["clean"])) for l in row) or 1
    conf = sum(l["conf"] * len(_alnum(l["clean"])) for l in row) / total
    return text, float(conf)


def read_plate_with_paddleocr(plate_crop, debug_tag=None):
    """
    Membaca teks plat dengan PaddleOCR. Pengganti read_plate_with_easyocr().
    Return: {"text", "confidence", "all_texts", "raw", "ms", "passes", "skipped", ...}
    """
    out = {"text": "", "confidence": 0.0, "all_texts": [], "raw": [], "ms": 0.0,
           "passes": 0, "skipped": None, "crop_w": 0, "crop_h": 0, "sharpness": 0.0}
    if paddle_engine is None:
        out["skipped"] = "engine_unavailable"
        return out
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
        out["skipped"] = "empty_crop"
        return out
    h, w = plate_crop.shape[:2]
    out["crop_w"], out["crop_h"] = w, h
    if h < 14 or w < 28:
        out["skipped"] = "too_small"
    else:
        gray0 = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        out["sharpness"] = float(cv2.Laplacian(gray0, cv2.CV_64F).var())
        if out["sharpness"] < PADDLE_MIN_SHARPNESS:
            out["skipped"] = "too_blurry"
    if out["skipped"]:
        print(f"[PADDLE] SKIP tag={debug_tag} reason={out['skipped']} crop={w}x{h} sharpness={out['sharpness']:.1f}")
        return out

    t0 = time.time()
    try:
        proc = _prepare_paddle_input(plate_crop)
        _save_debug_plate_crop(proc, debug_tag, "_p1")
        lines, _ = _paddle_pass(proc, "1")
        all_lines = list(lines)
        text, conf = _merge_main_plate_line(lines)
        passes = 1

        # Pass 2 (grayscale + CLAHE, invert jika plat gelap) HANYA bila pass 1 tidak menghasilkan teks yang jelas
        if len(_alnum(text)) < 4:
            g = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            if float(g.mean()) < 85.0:
                g = cv2.bitwise_not(g)
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
            proc2 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            _save_debug_plate_crop(proc2, debug_tag, "_p2")
            lines2, _ = _paddle_pass(proc2, "2")
            all_lines += lines2
            t2, c2 = _merge_main_plate_line(lines2)
            n1, n2 = len(_alnum(text)), len(_alnum(t2))
            if n2 > n1 or (n2 == n1 and c2 > conf):
                text, conf = t2, c2
            passes = 2
    except Exception as e:
        print(f"[PADDLE ERROR] tag={debug_tag}: {e}")
        out["skipped"] = f"error:{e}"
        return out

    out.update({
        "text": text,
        "confidence": float(conf) if text else 0.0,
        "all_texts": [l["clean"] for l in all_lines if l["clean"]],
        "raw": [(l["raw"], round(l["conf"], 3)) for l in all_lines],
        "ms": (time.time() - t0) * 1000.0,
        "passes": passes,
    })
    print(f"[PADDLE] FINAL tag={debug_tag} raw={out['raw']} normalized='{text}' "
          f"confidence={out['confidence']:.3f} passes={passes} ms={out['ms']:.1f}")
    return out


# ------------------------------------------------------------
# Normalisasi struktur plat Indonesia (GENERIK, bukan heuristik khusus EasyOCR / plat tertentu)
# Struktur: Prefix 1-2 huruf (kode wilayah) + 1-4 angka + Suffix 1-3 huruf. Format dinas: 523-07.
# Koreksi hanya berbasis posisi (angka di slot huruf / huruf di slot angka).
# ------------------------------------------------------------
_PREFIX_FIX = {'0': 'D', '8': 'B', '4': 'A', '5': 'S', '6': 'G', '2': 'Z'}
_SUFFIX_FIX = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
_DIGIT_FIX = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}


def _fit_slot(chars, fix_map, want_alpha):
    out, fixes = "", 0
    for ch in chars:
        ok = ch.isalpha() if want_alpha else ch.isdigit()
        if ok:
            out += ch
        elif ch in fix_map:
            out += fix_map[ch]
            fixes += 1
        else:
            return None
    return out, fixes


def normalize_indonesian_plate(raw_text):
    """
    Mengubah teks mentah OCR menjadi 'PREFIX DIGITS SUFFIX' bila struktur memungkinkan.
    Return dict: text, prefix, digits, suffix, valid, military, samsat_ok, corrections.
    Tidak mengarang karakter: bila struktur tidak cocok -> valid=False.
    """
    res = {"text": "", "prefix": "", "digits": "", "suffix": "", "valid": False,
           "military": False, "samsat_ok": False, "corrections": 0}
    if not raw_text:
        return res
    s = str(raw_text).upper().strip()
    m = re.fullmatch(r'\s*(\d{3,4})\s*-\s*(\d{2})\s*', s)
    if m:
        res.update({"text": f"{m.group(1)}-{m.group(2)}", "digits": m.group(1), "suffix": m.group(2),
                    "valid": True, "military": True})
        return res
    clean = re.sub(r'[^A-Z0-9]', '', s)
    n = len(clean)
    if n < 3 or n > 9:
        return res
    best = None
    for p_len in (1, 2):
        for d_len in (4, 3, 2, 1):
            s_len = n - p_len - d_len
            if not (1 <= s_len <= 3):
                continue
            pre = _fit_slot(clean[:p_len], _PREFIX_FIX, True)
            dig = _fit_slot(clean[p_len:p_len + d_len], _DIGIT_FIX, False)
            suf = _fit_slot(clean[p_len + d_len:], _SUFFIX_FIX, True)
            if pre is None or dig is None or suf is None:
                continue
            fixes = pre[1] + dig[1] + suf[1]
            cost = fixes + (0.0 if pre[0] in SAMSAT_PREFIXES else 1.5)
            if best is None or cost < best[0]:
                best = (cost, pre[0], dig[0], suf[0], fixes)
    if best is None:
        return res
    _, pre, dig, suf, fixes = best
    res.update({"text": f"{pre} {dig} {suf}", "prefix": pre, "digits": dig, "suffix": suf,
                "valid": True, "samsat_ok": pre in SAMSAT_PREFIXES, "corrections": fixes})
    return res


def fuse_plate_candidates(char_raw, char_conf, paddle_text, paddle_conf):
    """
    Menggabungkan dua sinyal independen. PaddleOCR adalah evidence utama:
    - Paddle valid & (char tidak valid / keduanya sepakat / conf Paddle tidak jauh di bawah char) -> Paddle
    - selain itu, jika char model valid -> char model
    confidence = confidence ASLI sumber terpilih (tidak dinaikkan).
    candidate_score = ukuran peringkat internal (conf + bonus kesepakatan/Samsat - penalti koreksi).
    """
    cn = normalize_indonesian_plate(char_raw)
    pn = normalize_indonesian_plate(paddle_text)
    agree = bool(cn["valid"] and pn["valid"] and cn["text"] == pn["text"])
    if pn["valid"] and (not cn["valid"] or agree or paddle_conf >= char_conf - PADDLE_PRIMARY_MARGIN):
        chosen, source, conf = pn, "paddle", float(paddle_conf)
    elif cn["valid"]:
        chosen, source, conf = cn, "char_model", float(char_conf)
    else:
        return {"text": "", "confidence": 0.0, "source": None, "agreement": False,
                "candidate_score": 0.0, "char_norm": cn, "paddle_norm": pn}
    score = conf + (0.15 if agree else 0.0) + (0.05 if chosen["samsat_ok"] else 0.0) - 0.05 * chosen["corrections"]
    return {"text": chosen["text"], "confidence": conf, "source": source, "agreement": agree,
            "candidate_score": round(max(0.0, min(1.0, score)), 3), "char_norm": cn, "paddle_norm": pn}


def ensemble_plate_reading(plate_crop, debug_tag=None):
    """
    Pipeline pembacaan satu crop plat:
      Character YOLO (di crop yang sudah di-sharpen)  ->  PaddleOCR (di crop asli)  ->  fusi kandidat.
    PaddleOCR SELALU dijalankan (engine OCR wajib), bukan fallback.
    """
    t_start = time.time()
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0 or plate_crop.shape[0] < 8 or plate_crop.shape[1] < 8:
        return {"final": "", "confidence": 0.0, "method": "invalid_crop"}

    raw_crop = plate_crop
    enhanced = enhance_moving_plate_crop(plate_crop)

    # 1. Character Model (sinyal karakter-level)
    t_c0 = time.time()
    char_raw, char_conf, line1_chars = read_plate_with_char_model(enhanced, conf=0.08)
    char_raw = char_raw.upper()
    c_clean = re.sub(r'[^A-Z0-9]', '', char_raw)
    paddle_input = raw_crop

    # 1b. Deskew hanya jika pembacaan karakter awal minim
    if char_conf < 0.45 or len(c_clean) < 4:
        deskewed_crop, skew_angle = deskew_plate(enhanced)
        if abs(skew_angle) >= 3.0:
            d_raw, d_conf, d_chars = read_plate_with_char_model(deskewed_crop, conf=0.08)
            if d_conf > char_conf and len(re.sub(r'[^A-Z0-9]', '', d_raw)) >= len(c_clean):
                print(f"[DEBUG] Koreksi Kemiringan Plat (Deskew): {skew_angle:+.1f} deg")
                char_raw, char_conf, line1_chars = d_raw.upper(), d_conf, d_chars
                hh, ww = raw_crop.shape[:2]
                M = cv2.getRotationMatrix2D((ww / 2.0, hh / 2.0), skew_angle, 1.0)
                paddle_input = cv2.warpAffine(raw_crop, M, (ww, hh), flags=cv2.INTER_CUBIC,
                                              borderMode=cv2.BORDER_REPLICATE)
    char_ms = (time.time() - t_c0) * 1000.0

    # 2. PaddleOCR (engine OCR utama)
    paddle = read_plate_with_paddleocr(paddle_input, debug_tag=debug_tag)
    paddle_text = (paddle.get("text") or "").upper()
    paddle_conf = float(paddle.get("confidence", 0.0) or 0.0)
    paddle_ms = float(paddle.get("ms", 0.0) or 0.0)

    # 3. Fusi dua sinyal independen
    fused = fuse_plate_candidates(char_raw, char_conf, paddle_text, paddle_conf)
    total_ms = (time.time() - t_start) * 1000.0

    perf_stats.add("character_model_ms", char_ms)
    if paddle.get("passes"):
        perf_stats.add("paddleocr_ms", paddle_ms)
    perf_stats.add("total_ocr_ms", total_ms)

    print(f"[CHAR] text='{char_raw}' conf={char_conf:.3f} ({len(line1_chars)} chars, {char_ms:.1f}ms)")
    print(f"[FUSE] final='{fused['text']}' source={fused['source']} conf={fused['confidence']:.3f} "
          f"agreement={fused['agreement']} candidate_score={fused['candidate_score']} total={total_ms:.1f}ms")

    method = {"paddle": "paddleocr_primary", "char_model": "char_model_only"}.get(fused["source"], "no_valid_reading")
    return {
        "final": fused["text"],
        "confidence": fused["confidence"],
        "method": method,
        "source": fused["source"],
        "agreement": fused["agreement"],
        "candidate_score": fused["candidate_score"],
        "char_raw": char_raw,
        "char_conf": char_conf,
        "paddle_text": paddle_text,
        "paddle_conf": paddle_conf,
        "paddle_all": paddle.get("all_texts", []),
        "paddle_skipped": paddle.get("skipped"),
        "char_ms": round(char_ms, 1),
        "paddle_ms": round(paddle_ms, 1),
        "total_ms": round(total_ms, 1),
    }


def classify_vehicle_indonesian(image, bbox, initial_vtype, v_conf, has_bus_det=False, body_hints=None):
    """
    Sistem klasifikasi kendaraan dan bodi terkalibrasi untuk lingkungan parkir Indonesia:
    - Mengintegrasikan petunjuk bodi dari model plate_detector_best.pt (Small Bus, Large Bus, Van, Hatchback, SUV, Truk).
    - Membedakan jenis kendaraan utama: car, motorcycle, truck, bus.
    - Menangani Isuzu Elf, HiAce, travel van -> masuk ke 'bus', 'Minibus' (bukan Truk).
    - Menangani bus besar / medium bus -> masuk ke 'bus', 'Bus'.
    - Menangani truk kargo komersial (Dump Truck, Box Truck, Canter, Dutro) -> 'truck', 'Truk'.
    - Menangani mobil penumpang: Sedan, Hatchback, SUV, MPV, Crossover, Pickup Truck.
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

    # Cocokkan petunjuk bodi dari model plate_detector_best.pt jika tersedia
    best_hint = None
    if body_hints:
        matching_hints = []
        for h in body_hints:
            hx1, hy1, hx2, hy2 = h["box"]
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0
            if (x1 - 30 <= hcx <= x2 + 30 and y1 - 30 <= hcy <= y2 + 30) or compute_iou(bbox, h["box"]) > 0.20:
                matching_hints.append(h)
        if matching_hints:
            matching_hints.sort(key=lambda x: -x["conf"])
            best_hint = matching_hints[0]

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

    # 1. EVALUASI PRIORITAS DARI MODEL PLATE_DETECTOR_BEST (BODY HINTS)
    if best_hint:
        hname = best_hint["name"]
        hconf = best_hint["conf"]
        if hname == "Small Bus" and hconf >= 0.35:
            return "bus", "Minibus", round(max(0.92, hconf), 3)
        if hname == "Large Bus" and hconf >= 0.35:
            return "bus", "Bus", round(max(0.95, hconf), 3)
        if hname in ["Medium Goods Vehicle"] and hconf >= 0.35 and p_suv < 0.20:
            return "truck", "Truk", round(max(0.95, hconf), 3)
        if hname == "Sports Utility Vehicle" and hconf >= 0.40:
            return "car", "SUV", round(max(0.92, hconf), 3)
        if hname == "Hatchback" and hconf >= 0.40 and aspect < 0.88:
            return "car", "Hatchback", round(max(0.90, hconf), 3)
        if hname == "Van" and hconf >= 0.30:
            return "car", "MPV", round(max(0.88, hconf), 3)
        if hname == "Sedan" and hconf >= 0.45:
            return "car", "Sedan", round(max(0.88, hconf), 3)

    # 2. EVALUASI TRUK KARGO / PICKUP (YOLO vehicle_model = 'truck')
    if initial_vtype == "truck":
        # Jika model bodi mendeteksi SUV kuat (> 40%), ini adalah SUV bukan truk!
        if p_suv >= 0.40 or (p_suv + p_crossover >= 0.50):
            return 'car', 'SUV', round(max(p_suv, 0.88), 3)
        if p_pickup >= 0.35:
            return 'truck', 'Pickup Truck', round(p_pickup, 3)
        is_elf = (has_bus_det and p_minibus >= 0.45) or (p_minibus >= 0.85 and v_conf < 0.70)
        if is_elf:
            return 'bus', 'Minibus', round(p_minibus, 3)
        return 'truck', 'Truk', round(v_conf, 3)

    # 3. EVALUASI BUS & MINIBUS (YOLO vehicle_model = 'bus')
    if initial_vtype == "bus":
        is_passenger_car = ((p_fastback >= 0.20 or p_sports >= 0.25 or p_sedan >= 0.20 or p_suv >= 0.20 or p_mpv >= 0.15 or p_wagon >= 0.15 or p_hatch >= 0.20) and p_minibus < 0.45)
        if not is_passenger_car:
            if p_minibus >= 0.45 and aspect < 0.95 and not (area_ratio >= 0.35 or w_ratio >= 0.65 or h_ratio >= 0.65):
                return 'bus', 'Minibus', round(p_minibus, 3)
            return 'bus', 'Bus', round(v_conf, 3)
        initial_vtype = "car"

    # 4. KENDARAAN MOBIL PENUMPANG (CAR)
    # A. SUV Kuat (seperti Toyota RAV4, Fortuner, Pajero Sport, HR-V, BMW iX)
    if p_suv >= 0.40 or (p_suv + p_crossover + p_pickup >= 0.50):
        score_suv = max(0.90, p_suv + p_crossover + p_pickup)
        return 'car', 'SUV', round(min(0.99, score_suv), 3)

    # B. Sedan / Fastback (Mercedes-Benz C/E-Class, Camry, Civic, Vios, Altis)
    p_sedan_total = p_sedan + (p_fastback * 0.95) + (p_conv * 0.75)
    if p_sedan_total >= 0.40 and (p_sedan >= 0.25 or p_fastback >= 0.30 or (p_fastback + p_conv >= 0.50)):
        score_sedan = max(0.88, p_sedan_total)
        return 'car', 'Sedan', round(min(0.99, score_sedan), 3)

    # C. Microcar / City Car Hatchback (Wuling Air EV, Brio)
    is_micro_hatch = (
        p_wagon < 0.05 and p_sedan < 0.08 and p_minibus < 0.15 and
        (
            (aspect >= 0.88 and (p_sports + p_fastback + p_hatch) >= 0.35 and w_ratio < 0.45) or
            (p_hatch >= 0.35 and aspect < 0.88)
        )
    )
    if is_micro_hatch:
        score_hatch = max(0.88, p_hatch + p_sports * 0.5)
        return 'car', 'Hatchback', round(min(0.99, score_hatch), 3)

    # D. Tall MPV / Minivan (Toyota Sienta, Alphard, Innova, Avanza, Calya, Sigra)
    # Sienta dan minivan kompak memiliki bodi jangkung (aspect >= 0.88 dan w_ratio >= 0.45)
    is_tall_mpv = (
        p_mpv >= 0.25 or
        (p_wagon >= 0.20 and aspect >= 0.72) or
        (aspect >= 0.88 and w_ratio >= 0.45 and (p_hatch >= 0.50 or p_wagon >= 0.10 or p_mpv >= 0.05))
    )
    if is_tall_mpv:
        score_mpv = max(0.88, p_mpv + (p_wagon * 0.7) + (p_hatch * 0.3 if aspect >= 0.88 else 0.0))
        return 'car', 'MPV', round(min(0.99, score_mpv), 3)

    # E. Sporty Crossover / EV SUV (seperti BMW iX, di mana p_sports sangat tinggi dan bodi jangkung aspect >= 0.75)
    if p_sports >= 0.45 and aspect >= 0.75 and p_sedan_total < 0.25:
        return 'car', 'SUV', round(max(0.88, p_sports), 3)

    # F. Sistem Skor Tertimbang Multikelas
    score_suv = p_suv + p_crossover + (p_pickup * 0.8) + (p_sports * 0.3 if aspect >= 0.75 else 0.0)
    score_sedan = p_sedan + (p_fastback * 0.9) + (p_conv * 0.7)
    score_mpv = p_mpv + (p_wagon * 0.8) + (p_minibus * 0.6)
    score_hatch = p_hatch + (p_wagon * 0.2)

    scores = {'SUV': score_suv, 'MPV': score_mpv, 'Sedan': score_sedan, 'Hatchback': score_hatch}
    best_cat = max(scores.items(), key=lambda kv: kv[1])[0]
    return 'car', best_cat, round(min(0.99, max(0.60, scores[best_cat])), 3)


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
# ============================================================
# FAST TEMPORAL CONFIRMATION & VEHICLE TRACKING (OPTIMIZED)
# Syarat konfirmasi Entry History:
# - Minimal 3 observasi plat yang cocok dengan confidence >= 80% dalam 5 observasi valid terakhir.
# - Memprioritaskan kecocokan berurutan (consecutive matches) ketika tersedia.
# - Mentolerir short gaps dari frame terlewat/low-confidence.
# - Mengunci data plat setelah terkonfirmasi (1x record, bebas duplikasi).
# ============================================================
class VehicleTrack:
    """
    Melacak dan mengonfirmasi kendaraan secara temporal stabil dengan Controlled Tracking & Lost Interest.
    State Machine: OUTSIDE_ROI -> ANALYZING -> CONFIRMED -> HISTORY_SAVED (atau LOST_INTEREST / DISCARDED)
    """
    def __init__(self, track_id, initial_det=None):
        self.track_id = track_id
        self.created_at = time.time()
        self.last_seen = time.time()
        self.frames = deque(maxlen=20)
        self.confirmed_data = None
        self.confirmation_score = 0.0
        self.history_saved = False
        self.is_locked = False
        self.last_bbox = None
        
        # Controlled Tracking & ROI State
        is_inside = initial_det.get("inside_interest_area", False) if initial_det else False
        self.inside_interest_area = is_inside
        self.ever_inside_roi = is_inside
        self.frames_outside_after_inside = 0
        self.lost_interest = False
        self.lost_interest_time = None
        self.status = "ANALYZING" if is_inside else "OUTSIDE_ROI"

        # Asynchronous OCR Management State
        self.ocr_pending = False
        self.last_ocr_job_id = 0
        self.last_processed_ocr_job_id = 0
        self.last_ocr_text = None
        self.last_ocr_conf = 0.0
        self.last_ocr_time = None

        # Cache body style to avoid re-classifying every frame
        self.cached_body_style = None
        self.cached_body_conf = 0.0
        self.cached_vtype = None
        self.best_plate_crop = None
        self.best_crop_quality = -1.0
        self.best_plate_conf = 0.0
        self.plate_readings = []

        # Latency Telemetry Timestamps
        self.first_good_ocr_time = None
        self.first_good_ocr_text = None
        self.first_good_ocr_conf = 0.0
        self.confirmation_time = None
        self.confirmation_reason = None
        self.history_saved_time = None

        # Candidate accumulation (persists across frames, never reset by noisy frame)
        self.candidate_counts = {}
        self.candidate_best_obs = {}
        self.best_candidate = None
        self.best_candidate_display = None
        self.best_candidate_conf = 0.0
        self.consecutive_plate = ""
        self.consecutive_count = 0
        self.valid_plate_observations = deque(maxlen=15)

        # PaddleOCR migration: state tambahan per track
        self.best_paddle_confidence = 0.0
        self.last_ocr_frame_id = None
        self.ocr_calls = 0
        self.roi_entry_time = self.created_at if is_inside else None
        self.first_plate_time = None
        self.first_ocr_result_time = None
        self.events_sent = set()
        self.lost_finalized = False
        self.bench_logged = False

        if initial_det:
            self.add_frame(initial_det)

    def add_frame(self, det):
        now = time.time()
        self.last_seen = now
        self.last_bbox = det.get("bbox")
        is_inside = det.get("inside_interest_area", False)
        self.inside_interest_area = is_inside

        if is_inside:
            self.ever_inside_roi = True
            if self.roi_entry_time is None:
                self.roi_entry_time = now
            self.frames_outside_after_inside = 0
            if self.status == "OUTSIDE_ROI":
                self.status = "ANALYZING"
        else:
            if self.ever_inside_roi:
                self.frames_outside_after_inside += 1
            elif self.status not in ["CONFIRMED", "HISTORY_SAVED"]:
                self.status = "OUTSIDE_ROI"

        p_crop = det.get("plate_crop")
        p_conf = det.get("plate_confidence", 0.0) or 0.0
        p_text = det.get("license_plate")

        # 1. Update best_plate_crop menggunakan metrik Crop Quality (HANYA jika kendaraan di dalam ROI)
        crop_q = 0.0
        if is_inside and p_crop is not None and getattr(p_crop, 'size', 0) > 0:
            crop_q = compute_crop_quality(p_crop, p_conf)
            if crop_q > self.best_crop_quality:
                self.best_plate_crop = p_crop
                self.best_crop_quality = crop_q
                self.best_plate_conf = p_conf

        # 2. Akumulasi pembacaan plat nomor valid HANYA saat kendaraan di dalam ROI
        #    dan HANYA dari hasil OCR baru (preview char-model / kandidat yang diputar ulang tidak dihitung)
        if is_inside and p_text and det.get("ocr_method") not in NON_EVIDENCE_OCR_METHODS:
            is_struct, p_type = is_valid_indonesian_plate_structure(p_text)
            p_clean = re.sub(r'[^A-Z0-9]', '', p_text.upper()) if p_text else ""

            if is_struct and p_conf >= 0.50:
                if self.first_good_ocr_time is None:
                    self.first_good_ocr_time = now
                    self.first_good_ocr_text = p_text
                    self.first_good_ocr_conf = p_conf
                    print(f"""[OCR GOOD]
track={self.track_id}
plate=\"{p_clean}\"
ocr_conf={p_conf:.2f}
timestamp={self.first_good_ocr_time:.3f}""")

                obs = {
                    "text": p_text,
                    "clean": p_clean,
                    "conf": p_conf,
                    "time": self.last_seen,
                    "crop_q": crop_q,
                    "struct": is_struct,
                    "type": p_type
                }
                self.valid_plate_observations.append(obs)
                self.plate_readings.append(obs)

                self.candidate_counts[p_clean] = self.candidate_counts.get(p_clean, 0) + 1
                if p_clean not in self.candidate_best_obs or p_conf > self.candidate_best_obs[p_clean]["conf"]:
                    self.candidate_best_obs[p_clean] = obs

                best_cln = max(self.candidate_counts, key=lambda c: self.candidate_counts[c] * 1.5 + self.candidate_best_obs[c]["conf"])
                self.best_candidate = best_cln
                self.best_candidate_display = self.candidate_best_obs[best_cln]["text"]
                self.best_candidate_conf = self.candidate_best_obs[best_cln]["conf"]

                if p_clean == self.consecutive_plate:
                    self.consecutive_count += 1
                else:
                    self.consecutive_plate = p_clean
                    self.consecutive_count = 1
            else:
                self.consecutive_plate = ""
                self.consecutive_count = 0

        self.frames.append({
            "time": self.last_seen,
            "vehicle_type": det.get("vehicle_type"),
            "v_conf": det.get("vehicle_confidence", 0.0) or 0.0,
            "body_style": det.get("body_style"),
            "body_conf": det.get("body_style_confidence", 0.0) or 0.0,
            "license_plate": p_text,
            "plate_conf": p_conf,
            "plate_crop": p_crop,
            "bbox": det.get("bbox"),
            "plate_bbox": det.get("plate_bbox"),
            "ocr_method": det.get("ocr_method"),
            "inside_interest_area": is_inside
        })

        # Jika sudah terkonfirmasi dan tersimpan di Entry History, kunci!
        if self.status in ["CONFIRMED", "HISTORY_SAVED"] or self.history_saved or self.is_locked:
            return

        if is_inside:
            self.status = "ANALYZING"
            self._evaluate_confirmation()
        else:
            if not self.ever_inside_roi:
                self.status = "OUTSIDE_ROI"

    def debug_snapshot(self, now=None, extra=None):
        """Snapshot debug telemetry untuk WebSocket / Latest Detection."""
        now = now or time.time()
        last_f = self.frames[-1] if self.frames else {}
        cand_count = self.candidate_counts.get(self.best_candidate, 0) if self.best_candidate else 0
        if self.lost_interest:
            ocr_stat = self.status if self.status in ("DISCARDED", "CONFIRMED", "HISTORY_SAVED") else "LOST_INTEREST"
        elif not self.inside_interest_area:
            ocr_stat = "OUTSIDE_ROI"
        elif self.status in ("CONFIRMED", "HISTORY_SAVED"):
            ocr_stat = "CONFIRMED"
        elif cand_count >= 2:
            ocr_stat = "GOOD_CANDIDATE"
        else:
            ocr_stat = "ANALYZING" if self.ever_inside_roi else "IDLE"
        snap = {
            "track_id": self.track_id,
            "vehicle_type": last_f.get("vehicle_type"),
            "body_style": last_f.get("body_style") or (self.confirmed_data or {}).get("body_style"),
            "vehicle_confidence": last_f.get("v_conf"),
            "inside_interest_area": self.inside_interest_area,
            "last_seen": round(self.last_seen, 3),
            "last_seen_age_ms": int(round(max(0.0, now - self.last_seen) * 1000)),
            "lost_interest": self.lost_interest,
            "status": self.status,
            "ocr_status": ocr_stat,
            "ocr_candidate": self.best_candidate_display,
            "ocr_confidence": round(self.best_candidate_conf, 3) if self.best_candidate_conf else None,
            "candidate_matches": cand_count,
            "plate_detected": bool(last_f.get("plate_bbox")),
            "best_candidate_score": self.get_best_candidate_score(),
            "temporal_consistency": round(self.get_temporal_consistency(), 2),
            "ocr_calls": self.ocr_calls,
            "bbox": self.last_bbox,
            "plate_bbox": last_f.get("plate_bbox"),
            "license_plate": (self.confirmed_data or {}).get("license_plate") or self.best_candidate_display,
            "plate_confidence": (self.confirmed_data or {}).get("plate_confidence") or (
                round(self.best_candidate_conf, 3) if self.best_candidate_conf else None
            ),
        }
        if extra:
            snap.update(extra)
        return snap

    def add_ocr_result(self, job_id, text, conf, crop_q, ocr_method="ensemble", frame_id=None, extra=None):
        """
        Mengevaluasi hasil OCR asinkron dan memperbarui tracking kandidat plat nomor.
        Menolak hasil OCR yang usang / job lama yang selesai lebih lambat dari job yang lebih baik.
        """
        now = time.time()
        is_stale_job = job_id < self.last_processed_ocr_job_id
        is_clearly_better = (crop_q > self.best_crop_quality * 1.20) or (
            conf > (self.best_candidate_conf or 0.0) + 0.10
        )
        if is_stale_job and not is_clearly_better:
            print(f"[OCR REJECT] Stale job {job_id} < {self.last_processed_ocr_job_id} for track {self.track_id}")
            return None
        if self.best_candidate_conf and conf < (self.best_candidate_conf - 0.08) and crop_q < self.best_crop_quality:
            print(f"[OCR REJECT] Weaker result conf={conf:.2f} < best={self.best_candidate_conf:.2f} for track {self.track_id}")
            self.last_processed_ocr_job_id = max(self.last_processed_ocr_job_id, job_id)
            if frame_id is not None:
                self.last_ocr_frame_id = frame_id
            return {
                "track_id": self.track_id,
                "ocr_candidate": self.best_candidate_display or text,
                "ocr_confidence": self.best_candidate_conf,
                "candidate_matches": self.candidate_counts.get(self.best_candidate, 0) if self.best_candidate else 0,
                "status": self.status,
                "is_confirmed": self.status in ["CONFIRMED", "HISTORY_SAVED"]
            }

        self.last_processed_ocr_job_id = max(self.last_processed_ocr_job_id, job_id)
        if frame_id is not None:
            self.last_ocr_frame_id = frame_id
        self.best_paddle_confidence = max(self.best_paddle_confidence, float((extra or {}).get("paddle_conf") or 0.0))
        self.last_ocr_text = text
        self.last_ocr_conf = conf
        self.last_ocr_time = now
        if crop_q > self.best_crop_quality:
            self.best_crop_quality = crop_q

        is_struct, p_type = is_valid_indonesian_plate_structure(text)
        p_clean = re.sub(r'[^A-Z0-9]', '', text.upper()) if text else ""

        if is_struct and conf >= 0.50:
            if self.first_good_ocr_time is None:
                self.first_good_ocr_time = now
                self.first_good_ocr_text = text
                self.first_good_ocr_conf = conf
                print(f"""[OCR GOOD]
track={self.track_id}
plate=\"{p_clean}\"
ocr_conf={conf:.2f}
timestamp={self.first_good_ocr_time:.3f}""")

            obs = {
                "text": text,
                "clean": p_clean,
                "conf": conf,
                "time": now,
                "crop_q": crop_q,
                "struct": is_struct,
                "type": p_type,
                "method": ocr_method,
                "score": float((extra or {}).get("candidate_score") or conf),
                "source": (extra or {}).get("source"),
                "frame_id": frame_id
            }
            self.valid_plate_observations.append(obs)
            self.plate_readings.append(obs)

            self.candidate_counts[p_clean] = self.candidate_counts.get(p_clean, 0) + 1
            if p_clean not in self.candidate_best_obs or conf > self.candidate_best_obs[p_clean]["conf"]:
                self.candidate_best_obs[p_clean] = obs

            best_cln = max(self.candidate_counts, key=lambda c: self.candidate_counts[c] * 1.5 + self.candidate_best_obs[c]["conf"])
            self.best_candidate = best_cln
            self.best_candidate_display = self.candidate_best_obs[best_cln]["text"]
            self.best_candidate_conf = self.candidate_best_obs[best_cln]["conf"]

            if p_clean == self.consecutive_plate:
                self.consecutive_count += 1
            else:
                self.consecutive_plate = p_clean
                self.consecutive_count = 1

            if not self.is_locked and self.status not in ["CONFIRMED", "HISTORY_SAVED"]:
                self._evaluate_confirmation()
        else:
            self.consecutive_plate = ""
            self.consecutive_count = 0

        return {
            "track_id": self.track_id,
            "ocr_candidate": self.best_candidate_display or text,
            "ocr_confidence": self.best_candidate_conf if self.best_candidate else conf,
            "candidate_matches": self.candidate_counts.get(self.best_candidate, 1) if self.best_candidate else 0,
            "status": self.status,
            "is_confirmed": self.status in ["CONFIRMED", "HISTORY_SAVED"]
        }

    def check_lost_interest(self, now=None):
        """
        Memeriksa apakah kendaraan telah meninggalkan Detection Area atau tidak terlihat melebihi batas waktu (Lost Interest).
        Saat LOST: tidak ada job OCR baru yang dibuat. Job PaddleOCR yang sudah berjalan dibiarkan selesai
        (maks LOST_OCR_GRACE_SEC) sebelum kandidat terbaik difinalisasi / dibuang.
        - Finalisasi kandidat terbaik jika memenuhi syarat konfirmasi (Path A atau Path B).
        - Jika tidak memenuhi syarat: buang kandidat (DISCARDED) agar riwayat palsu tidak pernah tercatat.
        """
        if now is None:
            now = time.time()
        if self.lost_interest:
            if self.lost_finalized:
                return self.status
            return self._finalize_lost(now)

        time_since_seen = now - self.last_seen
        is_time_lost = time_since_seen > LOST_INTEREST_TIMEOUT_SEC
        is_left_roi = (self.ever_inside_roi and self.frames_outside_after_inside >= LOST_INTEREST_OUTSIDE_FRAMES)

        if is_time_lost or is_left_roi:
            self.lost_interest = True
            self.lost_interest_time = now
            print(f"[LOST INTEREST] Track {self.track_id} ditandai LOST (age={time_since_seen:.2f}s, outside_frames={self.frames_outside_after_inside})")
            return self._finalize_lost(now)

        return self.status

    def _finalize_lost(self, now):
        if self.status in ["CONFIRMED", "HISTORY_SAVED"] or self.history_saved:
            self.lost_finalized = True
            return self.status

        # Biarkan job PaddleOCR yang sudah berjalan selesai (dibatasi grace period)
        if self.ocr_pending and (now - (self.lost_interest_time or now)) < LOST_OCR_GRACE_SEC:
            return self.status

        self.lost_finalized = True
        best_cln = self.best_candidate
        if best_cln and best_cln in self.candidate_best_obs:
            best_obs = self.candidate_best_obs[best_cln]
            best_conf = best_obs["conf"]
            best_count = self.candidate_counts.get(best_cln, 0)
            is_struct = best_obs["struct"]
            p_type = best_obs["type"]

            can_finalize = (is_struct and best_count >= 2 and best_conf >= 0.70) or \
                           (is_struct and best_conf >= 0.85 and (len(best_cln) >= 5 or p_type == "military"))

            if can_finalize:
                reason = f"LOST_INTEREST_FINALIZED (count={best_count}, conf={best_conf:.2f})"
                print(f"[LOST INTEREST] Kandidat '{best_cln}' memenuhi syarat konfirmasi -> Finalisasi ke Entry History!")
                self.confirmation_time = now
                self.confirmation_reason = reason
                self._finalize_confirmation(best_obs["text"], best_conf, matching_count=best_count, reason=reason)
                return "CONFIRMED"
            else:
                print(f"[LOST INTEREST] Kandidat '{best_cln}' TIDAK memenuhi syarat (count={best_count}, conf={best_conf:.2f}) -> Dibuang tanpa masuk riwayat!")
                self.status = "DISCARDED"
                return "DISCARDED"
        else:
            self.status = "DISCARDED"
            return "DISCARDED"

    def _evaluate_confirmation(self):
        if not self.candidate_counts:
            return

        now = time.time()
        best_cln = self.best_candidate
        if not best_cln or best_cln not in self.candidate_best_obs:
            return

        best_obs = self.candidate_best_obs[best_cln]
        best_plate = best_obs["text"]
        best_conf = best_obs["conf"]
        best_count = self.candidate_counts[best_cln]
        is_struct = best_obs["struct"]
        p_type = best_obs["type"]

        confirmed = False
        reason = None

        # ============================================================
        # PATH A — HIGH CONFIDENCE FAST CONFIRMATION
        # ============================================================
        # A1: Single observation with very strong evidence (conf >= 0.85 and valid structure)
        if is_struct and best_conf >= 0.85 and (len(best_cln) >= 5 or p_type == "military"):
            confirmed = True
            reason = f"PATH_A_STRONG_SINGLE_OBSERVATION (conf={best_conf:.2f})"

        # A2: Two matching observations with solid confidence (>= 0.75)
        elif is_struct and best_count >= 2 and best_conf >= 0.75:
            confirmed = True
            reason = f"PATH_A_TWO_FRAME_MATCH (count={best_count}, conf={best_conf:.2f})"

        # ============================================================
        # PATH B — UNCERTAIN / MEDIUM OCR TEMPORAL CONFIRMATION
        # ============================================================
        # B1: Consecutive streak >= 3 with confidence >= 0.70
        elif self.consecutive_count >= 3 and self.consecutive_plate == best_cln and best_conf >= 0.70:
            confirmed = True
            reason = f"PATH_B_CONSECUTIVE_STREAK (streak={self.consecutive_count}, conf={best_conf:.2f})"

        # B2: Temporal consensus: >= 3 matching occurrences in observations with confidence >= 0.65
        elif is_struct and best_count >= 3 and best_conf >= 0.65:
            confirmed = True
            reason = f"PATH_B_TEMPORAL_CONSENSUS (matches={best_count}, conf={best_conf:.2f})"

        if confirmed:
            self.confirmation_time = now
            self.confirmation_reason = reason
            dt_from_good = ((now - self.first_good_ocr_time) * 1000.0) if self.first_good_ocr_time else 0.0
            print(f"""[CONFIRMATION]
track={self.track_id}
plate=\"{best_cln}\"
reason={reason}
time_from_good_ocr_to_confirmation_ms={dt_from_good:.1f}""")
            self._finalize_confirmation(best_plate, best_conf, matching_count=best_count, reason=reason)

    def _finalize_confirmation(self, stable_plate, stable_conf, matching_count=1, reason=""):
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

        # Frame terbaik untuk snapshot / bounding box
        best_f = max(self.frames, key=lambda f: (f["plate_conf"] if f["plate_conf"] else 0.0) + f["body_conf"])

        self.confirmed_data = {
            "track_id": self.track_id,
            "vehicle_type": best_type,
            "body_style": best_style,
            "body_style_confidence": round(avg_style_conf, 3) if avg_style_conf else None,
            "license_plate": stable_plate,
            "plate_confidence": round(stable_conf, 3),
            "consistency": round(consistency_score, 2),
            "bbox": best_f["bbox"],
            "plate_bbox": best_f["plate_bbox"],
            "ocr_method": best_f.get("ocr_method") or "fast_path",
            "confirmation_reason": reason,
            "confirmation_time": self.confirmation_time
        }
        self.confirmation_score = round(consistency_score, 2)
        self.status = "CONFIRMED"
        self.is_locked = True

    def get_current_stability_count(self):
        """Mengembalikan jumlah observasi plat yang cocok saat ini (untuk progress UI 1/3, 2/3, dst)."""
        if self.status in ["CONFIRMED", "HISTORY_SAVED"] or self.history_saved:
            return 3
        if self.best_candidate and self.best_candidate in self.candidate_counts:
            return min(self.candidate_counts[self.best_candidate], 3)
        if self.consecutive_count > 0:
            return min(self.consecutive_count, 3)
        return 1 if self.frames else 0

    def get_best_candidate_score(self):
        if self.best_candidate and self.best_candidate in self.candidate_best_obs:
            return round(self.candidate_best_obs[self.best_candidate].get("score", self.best_candidate_conf), 3)
        return None

    def get_temporal_consistency(self):
        total = sum(self.candidate_counts.values())
        if not total or not self.best_candidate:
            return 0.0
        return self.candidate_counts.get(self.best_candidate, 0) / float(total)

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
    Mengelola multi-object tracking, controlled tracking, dan konfirmasi temporal kendaraan.
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

    def get_or_create_track(self, raw_tid, bbox, img_w=1920, img_h=1080):
        """Ambil atau buat track sebelum plate/OCR agar job asinkron tidak kehilangan target."""
        with self.lock:
            if raw_tid is not None:
                tid = raw_tid
            else:
                matched_id = self._match_track(bbox, img_w=img_w, img_h=img_h)
                if matched_id is not None:
                    tid = matched_id
                else:
                    tid = self.next_fallback_id
                    self.next_fallback_id += 1
            if tid not in self.tracks:
                self.tracks[tid] = VehicleTrack(tid, None)
                if bbox:
                    self.tracks[tid].last_bbox = bbox
            return tid, self.tracks[tid]

    def check_lost_tracks(self):
        """Mengevaluasi seluruh track aktif untuk Lost Interest dan memfinalisasi kandidat yang valid."""
        now = time.time()
        newly_confirmed_tracks = []
        newly_lost_tracks = []
        with self.lock:
            stale_ids = []
            for tid, trk in list(self.tracks.items()):
                was_lost = trk.lost_interest
                prev_status = trk.status
                new_status = trk.check_lost_interest(now)
                if not was_lost and trk.lost_interest:
                    newly_lost_tracks.append(trk)
                if prev_status != "CONFIRMED" and new_status == "CONFIRMED":
                    newly_confirmed_tracks.append(trk)

                # Hapus track yang sudah lost > 10 detik agar memori bersih
                if trk.lost_interest and (now - trk.last_seen) > 10.0:
                    stale_ids.append(tid)
            for tid in stale_ids:
                del self.tracks[tid]
        return newly_confirmed_tracks, newly_lost_tracks

    def _match_track(self, bbox, curr_plate=None, img_w=1920, img_h=1080, iou_thresh=0.20, max_center_dist_ratio=0.18):
        if not bbox:
            return None
        now = time.time()
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        best_id = None
        best_score = -1.0

        curr_clean = re.sub(r'[^A-Z0-9]', '', curr_plate.upper()) if curr_plate else ""

        for tid, trk in self.tracks.items():
            if (now - trk.last_seen) > 3.0:
                continue

            # Jika track lama sudah punya plat nomor terkonfirmasi dan plat saat ini berbeda jelas, jangan match!
            if trk.confirmed_data and trk.confirmed_data.get("license_plate"):
                trk_clean = re.sub(r'[^A-Z0-9]', '', trk.confirmed_data["license_plate"].upper())
                if len(trk_clean) >= 5 and len(curr_clean) >= 5 and trk_clean != curr_clean:
                    continue

            if trk.last_bbox:
                lx1, ly1, lx2, ly2 = trk.last_bbox
                lcx = (lx1 + lx2) / 2.0
                lcy = (ly1 + ly2) / 2.0

                iou = compute_iou(bbox, trk.last_bbox)
                dx = abs(cx - lcx) / max(1, img_w)
                dy = abs(cy - lcy) / max(1, img_h)
                dist = (dx**2 + dy**2)**0.5

                if iou >= iou_thresh or dist <= max_center_dist_ratio:
                    score = iou + (1.0 - min(1.0, dist / max_center_dist_ratio))
                    if score > best_score:
                        best_score = score
                        best_id = tid
        return best_id

    def resolve_track_id(self, raw_tid, bbox, img_w=1920, img_h=1080):
        """Menghasilkan atau memetakan track ID sebelum inferensi lanjutan."""
        with self.lock:
            if raw_tid is not None:
                return raw_tid
            matched_id = self._match_track(bbox, img_w=img_w, img_h=img_h)
            if matched_id is not None:
                return matched_id
            tid = self.next_fallback_id
            self.next_fallback_id += 1
            return tid

    def update(self, detections, is_stream=False):
        with self.lock:
            now = time.time()
            stale_ids = [tid for tid, trk in self.tracks.items() if (now - trk.last_seen) > 8.0 and trk.lost_interest]
            for tid in stale_ids:
                del self.tracks[tid]

            if not detections:
                return []

            # Jika single photo upload manual (bukan live stream)
            if not is_stream:
                for det in detections:
                    p_text = det.get("license_plate")
                    p_conf = det.get("plate_confidence", 0.0) or 0.0
                    is_valid_struct, p_type = is_valid_indonesian_plate_structure(p_text)
                    p_clean = re.sub(r'[^A-Z0-9]', '', p_text.upper()) if p_text else ""

                    det["track_id"] = 1
                    det["inside_interest_area"] = True
                    det["lost_interest"] = False
                    det["last_seen"] = now
                    det["last_seen_age_ms"] = 0
                    det["ocr_candidate"] = p_text
                    det["ocr_confidence"] = p_conf
                    det["candidate_confidence"] = p_conf
                    det["candidate_matches"] = 1
                    det["plate_detected"] = bool(det.get("plate_bbox") or det.get("plate_detected"))

                    # SINGLE PHOTO FAST PATH: Valid Indonesian plate structure with confidence >= 0.65
                    if is_valid_struct and p_conf >= 0.65:
                        now_ts = time.time()
                        det["status"] = "CONFIRMED"
                        det["ocr_status"] = "CONFIRMED"
                        det["consistency"] = 1.0
                        det["is_newly_confirmed"] = True
                        det["confirmation_time"] = now_ts
                        print(f"""[CONFIRMATION]
track=1
plate=\"{p_clean}\"
reason=SINGLE_PHOTO_FAST_PATH (conf={p_conf:.2f})
time_from_good_ocr_to_confirmation_ms=0.0""")
                    else:
                        det["status"] = "ANALYZING"
                        det["ocr_status"] = "ANALYZING"
                        det["consistency"] = 0.5
                        det["is_newly_confirmed"] = False
                        det["analyzing_frame_count"] = 1
                        det["analyzing_max_frames"] = 3
                    det.pop("plate_crop", None)
                return detections

            # Mode Stream / Live CCTV
            updated_detections = []
            for det in detections:
                raw_tid = det.get("track_id")
                bbox = det.get("bbox")
                iw = det.get("image_width", 1920)
                ih = det.get("image_height", 1080)
                curr_plate_pre = det.get("license_plate")

                if raw_tid is not None:
                    tid = raw_tid
                else:
                    matched_id = self._match_track(bbox, curr_plate=curr_plate_pre, img_w=iw, img_h=ih)
                    if matched_id is not None:
                        tid = matched_id
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

                # Pasang status temporal konfirmasi & telemetry ke detection object
                det["inside_interest_area"] = track.inside_interest_area
                det["lost_interest"] = track.lost_interest
                det["last_seen"] = round(track.last_seen, 3)
                det["last_seen_age_ms"] = int(round(max(0.0, now - track.last_seen) * 1000))
                det["ocr_candidate"] = track.best_candidate_display
                det["candidate_confidence"] = round(track.best_candidate_conf, 3) if track.best_candidate_conf else None
                det["candidate_matches"] = track.candidate_counts.get(track.best_candidate, 0) if track.best_candidate else 0
                det["plate_detected"] = bool(det.get("plate_bbox") or det.get("plate_detected"))

                if track.lost_interest:
                    det["status"] = track.status if track.status in ("DISCARDED", "CONFIRMED", "HISTORY_SAVED") else "LOST_INTEREST"
                    det["ocr_status"] = "LOST_INTEREST" if det["status"] == "LOST_INTEREST" else det["status"]
                    det["is_newly_confirmed"] = False
                    if track.confirmed_data and not track.history_saved:
                        det["license_plate"] = track.confirmed_data.get("license_plate")
                        det["ocr_confidence"] = track.confirmed_data.get("plate_confidence")
                    elif track.best_candidate_display:
                        det["license_plate"] = track.best_candidate_display
                        det["ocr_confidence"] = round(track.best_candidate_conf, 3) if track.best_candidate_conf else None
                    det.pop("plate_crop", None)
                    updated_detections.append(det)
                    continue

                if not track.inside_interest_area:
                    det["status"] = "OUTSIDE_ROI"
                    det["ocr_status"] = "OUTSIDE_ROI"
                    det["license_plate"] = None
                    det["plate_bbox"] = None
                    det["plate_confidence"] = None
                    det["ocr_confidence"] = None
                    det["is_newly_confirmed"] = False
                elif track.history_saved:
                    det["status"] = "HISTORY_SAVED"
                    det["ocr_status"] = "CONFIRMED"
                    det["is_newly_confirmed"] = False
                    det["consistency"] = track.confirmation_score

                    if track.confirmed_data:
                        det["body_style"] = track.confirmed_data["body_style"]
                        det["body_style_confidence"] = track.confirmed_data["body_style_confidence"]
                        det["license_plate"] = track.confirmed_data["license_plate"]
                        det["plate_confidence"] = track.confirmed_data["plate_confidence"]
                        det["ocr_confidence"] = track.confirmed_data["plate_confidence"]
                        det["vehicle_type"] = track.confirmed_data["vehicle_type"]
                elif track.status == "CONFIRMED":
                    det["status"] = "CONFIRMED"
                    det["ocr_status"] = "CONFIRMED"
                    det["is_newly_confirmed"] = True
                    det["confirmation_time"] = track.confirmation_time
                    track.status = "HISTORY_SAVED"
                    track.history_saved = True
                    track.is_locked = True
                    det["consistency"] = track.confirmation_score

                    if track.confirmed_data:
                        det["body_style"] = track.confirmed_data["body_style"]
                        det["body_style_confidence"] = track.confirmed_data["body_style_confidence"]
                        det["license_plate"] = track.confirmed_data["license_plate"]
                        det["plate_confidence"] = track.confirmed_data["plate_confidence"]
                        det["ocr_confidence"] = track.confirmed_data["plate_confidence"]
                        det["vehicle_type"] = track.confirmed_data["vehicle_type"]
                else:
                    det["status"] = "ANALYZING"
                    cand_count = track.candidate_counts.get(track.best_candidate, 0) if track.best_candidate else 0
                    det["ocr_status"] = "GOOD_CANDIDATE" if cand_count >= 2 else "ANALYZING"
                    det["is_newly_confirmed"] = False
                    det["consistency"] = track.get_current_consistency()
                    if track.best_candidate_display:
                        det["license_plate"] = track.best_candidate_display
                    if track.best_candidate_conf:
                        det["ocr_confidence"] = round(track.best_candidate_conf, 3)
                    det["analyzing_frame_count"] = max(1, track.get_current_stability_count())
                    det["analyzing_max_frames"] = 3

                det.pop("plate_crop", None)
                updated_detections.append(det)

            return updated_detections


# ============================================================
# TELEMETRI PERFORMA, EVENT ALPR, DAN STATE ALPR
# ============================================================
class PerfStats:
    """Pengukur latency bergulir (500 sampel terakhir) + ringkasan per kendaraan. Dibaca via GET /api/perf."""
    KEYS = ("vehicle_detection_ms", "plate_detection_ms", "character_model_ms",
            "paddleocr_ms", "total_ocr_ms", "frame_age_ms")

    def __init__(self):
        self.lock = threading.Lock()
        self.series = {k: deque(maxlen=500) for k in self.KEYS}
        self.completed = deque(maxlen=200)

    def add(self, key, value):
        if value is None or key not in self.series:
            return
        with self.lock:
            self.series[key].append(float(value))

    def summary(self):
        out = {}
        with self.lock:
            for k, v in self.series.items():
                if not v:
                    out[k] = {"n": 0}
                    continue
                arr = sorted(v)
                out[k] = {
                    "n": len(arr),
                    "mean": round(sum(arr) / len(arr), 1),
                    "p50": round(arr[len(arr) // 2], 1),
                    "p95": round(arr[min(len(arr) - 1, int(len(arr) * 0.95))], 1),
                }
            comp = list(self.completed)
        calls = [c["ocr_calls"] for c in comp]
        out["ocr_calls_per_vehicle"] = {"n": len(calls), "mean": round(sum(calls) / len(calls), 2)} if calls else {"n": 0}
        for k in ("time_to_first_plate_ms", "time_to_first_ocr_ms", "time_to_confirmation_ms"):
            vals = [c[k] for c in comp if c.get(k) is not None]
            out[k] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 1)} if vals else {"n": 0}
        return out


perf_stats = PerfStats()


def log_track_bench(trk, outcome):
    """Ringkasan latency per kendaraan (dihitung dari saat kendaraan masuk ROI)."""
    if trk is None or trk.bench_logged:
        return
    trk.bench_logged = True
    base = trk.roi_entry_time

    def d(t):
        return round((t - base) * 1000.0, 1) if (t and base) else None

    summary = {
        "track_id": trk.track_id,
        "outcome": outcome,
        "ocr_calls": trk.ocr_calls,
        "time_to_first_plate_ms": d(trk.first_plate_time),
        "time_to_first_ocr_ms": d(trk.first_ocr_result_time),
        "time_to_confirmation_ms": d(trk.confirmation_time),
        "best_candidate": trk.best_candidate_display,
    }
    perf_stats.completed.append(summary)
    print(f"[BENCH] {summary}")


def emit_track_event_once(trk, name, **extra):
    """Siarkan event ALPR lewat WebSocket sekali per track (vehicle_detected, entered_roi, plate_detected, ...)."""
    if trk is None or name in trk.events_sent:
        return
    trk.events_sent.add(name)
    ws_broadcaster.broadcast({"type": "alpr_event", "event": name, "track_id": trk.track_id,
                              "timestamp": time.time(), **extra})


def handle_lost_tracks(newly_lost):
    """Event lost_interest + log bench untuk track yang sudah final. Dipanggil di luar lock manager."""
    for trk in newly_lost:
        emit_track_event_once(trk, "lost_interest", status=trk.status)
    with confirmation_manager.lock:
        done = [t for t in confirmation_manager.tracks.values()
                if t.lost_interest and t.lost_finalized and not t.bench_logged]
    for t in done:
        log_track_bench(t, t.status)


def compute_alpr_state(detections):
    """
    State ALPR untuk UI (terpisah dari status WebSocket & latency):
    WAITING_FOR_VEHICLE | ALPR_ACTIVE | OCR_ANALYZING | CONFIRMED | LOST_INTEREST
    """
    active = [d for d in detections if d.get("inside_interest_area") and not d.get("lost_interest")]
    if any(d.get("status") in ("CONFIRMED", "HISTORY_SAVED") for d in active):
        return "CONFIRMED"
    if any(d.get("status") == "ANALYZING" and d.get("plate_detected") for d in active):
        return "OCR_ANALYZING"
    if active:
        return "ALPR_ACTIVE"
    now = time.time()
    with confirmation_manager.lock:
        recent_lost = any(t.lost_interest and t.lost_interest_time and (now - t.lost_interest_time) <= 3.0
                          for t in confirmation_manager.tracks.values())
    return "LOST_INTEREST" if recent_lost else "WAITING_FOR_VEHICLE"


def _async_ocr_task(job_id, track_id, crop, crop_q, submit_time, meta=None):
    """
    Worker task untuk asynchronous OCR (Character YOLO + PaddleOCR) di thread pool.
    Setiap job membawa track_id, frame_id, job_id, dan capture_timestamp (meta).
    Hasil job lama tidak boleh menimpa kandidat yang lebih baik (dijaga di VehicleTrack.add_ocr_result).
    """
    meta = meta or {}
    frame_id = meta.get("frame_id")
    try:
        t_job0 = time.time()
        queue_ms = (t_job0 - submit_time) * 1000.0
        res = ensemble_plate_reading(crop, debug_tag=f"t{track_id}_j{job_id}")
        text = res.get("final") or ""
        conf = float(res.get("confidence", 0.0) or 0.0)
        method = res.get("method", "ensemble")

        is_newly_confirmed = False
        confirmed_data = None
        update_info = None
        track = None
        temporal = 0.0
        inside_roi = None
        lost_flag = None
        best_display = None
        cur_status = None

        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if not track:
                return
            track.ocr_pending = False
            track.ocr_calls += 1
            if track.lost_interest and track.lost_finalized:
                print(f"[OCR LATE] Job {job_id} untuk track {track_id} selesai setelah finalisasi LOST -> diabaikan")
                return
            if text and track.first_ocr_result_time is None:
                track.first_ocr_result_time = time.time()
            prev_status = track.status
            update_info = track.add_ocr_result(job_id, text, conf, crop_q, ocr_method=method,
                                               frame_id=frame_id, extra=res)
            temporal = track.get_temporal_consistency()
            inside_roi, lost_flag = track.inside_interest_area, track.lost_interest
            best_display = track.best_candidate_display
            if prev_status != "CONFIRMED" and track.status == "CONFIRMED":
                is_newly_confirmed = True
                confirmed_data = dict(track.confirmed_data) if track.confirmed_data else None
                track.history_saved = True
                track.status = "HISTORY_SAVED"
                track.is_locked = True
            cur_status = track.status

        print(f"[TRACK] track_id={track_id} inside_roi={inside_roi} lost_interest={lost_flag} job={job_id} "
              f"frame={frame_id} queue_ms={queue_ms:.0f}")
        print(f"[RESULT] current_candidate='{text}' best_candidate='{best_display}' "
              f"confirmation_status={cur_status} paddle='{res.get('paddle_text')}' char='{res.get('char_raw')}'")

        # Siarkan pembaruan OCR langsung ke WebSocket tanpa menunggu konfirmasi
        if update_info:
            cand_count = update_info.get("candidate_matches", 0)
            ocr_stat = "CONFIRMED" if update_info["is_confirmed"] else ("GOOD_CANDIDATE" if cand_count >= 2 else "ANALYZING")
            ws_broadcaster.broadcast({
                "type": "ocr_update",
                "track_id": track_id,
                "frame_id": frame_id,
                "job_id": job_id,
                "license_plate": update_info["ocr_candidate"],
                "ocr_candidate": update_info["ocr_candidate"],
                "ocr_confidence": round(update_info["ocr_confidence"], 3) if update_info.get("ocr_confidence") else None,
                "candidate_matches": cand_count,
                "ocr_status": ocr_stat,
                "status": update_info["status"],
                "paddle_text": res.get("paddle_text"),
                "paddle_confidence": round(res.get("paddle_conf", 0.0) or 0.0, 3),
                "character_model_text": res.get("char_raw"),
                "character_model_confidence": round(res.get("char_conf", 0.0) or 0.0, 3),
                "candidate_score": res.get("candidate_score"),
                "temporal_consistency": round(temporal, 2),
                "lost_interest": bool(lost_flag),
                "inside_interest_area": True,
                "timestamp": time.time()
            })

        if is_newly_confirmed and confirmed_data:
            rec = save_parking_record(confirmed_data)
            log_track_bench(track, "CONFIRMED")
    except Exception as e:
        print(f"[ASYNC OCR ERROR] Track {track_id} job {job_id}: {e}")
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if track:
                track.ocr_pending = False



confirmation_manager = VehicleConfirmationManager()


def run_anpr(image_input, vehicle_conf=None, motorcycle_conf=None, plate_conf=None, single_vehicle_mode=True, is_stream=False,
             frame_id=None, capture_ts=None):
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

    # 1. YOLO VEHICLE INFERENCE (plate detection is gated by interest area)
    t_yolo_0 = time.time()
    if is_stream:
        fut_v = ai_pool.submit(vehicle_model.track, img, persist=True, tracker="bytetrack.yaml",
                               conf=m_conf_thresh, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    else:
        fut_v = ai_pool.submit(vehicle_model.predict, img, conf=m_conf_thresh,
                               imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    vdet = fut_v.result()[0]
    t_vehicle = time.time() - t_yolo_0

    # 2. EKSTRAKSI KANDIDAT KENDARAAN
    candidates = []
    for box in vdet.boxes:
        v_cls = int(box.cls[0])
        v_conf = float(box.conf[0])
        track_id = int(box.id[0]) if (box.id is not None) else None
        vehicle_type = map_vehicle_class_name(v_cls, vehicle_model)
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

    global_plates = []
    body_hints = []

    # 2b. FILTER & FOKUS KENDARAAN DI DEPAN KAMERA (Front-of-Camera Priority)
    scored_candidates = []
    if candidates:
        for c in candidates:
            s = get_front_of_camera_score(c, global_plates, iw, ih)
            if s > 0:
                c["front_score"] = s
                inside, overlap = is_vehicle_inside_roi(c["x1"], c["y1"], c["x2"], c["y2"], iw, ih)
                c["_inside_roi"] = inside
                c["_roi_overlap"] = overlap
                scored_candidates.append(c)
        scored_candidates.sort(key=lambda c: (-int(c.get("_inside_roi", False)), -c["front_score"]))

        if single_vehicle_mode and not is_stream:
            candidates = [scored_candidates[0]] if scored_candidates else []
        else:
            candidates = scored_candidates

    any_inside_roi = any(c.get("_inside_roi") for c in candidates)

    # Plate detection/OCR hanya jika ada kendaraan di dalam Detection Area
    # (foto tunggal tetap menjalankan detektor plat seperti jalur last-known-good)
    plate_stage_ran = bool(any_inside_roi or not is_stream)
    t_plate_total = 0.0
    t_plate_block0 = time.time()
    if any_inside_roi or not is_stream:
        pdet_global = plate_detector.predict(img, conf=p_conf_thresh, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
        for pbox in pdet_global.boxes:
            cls_id = int(pbox.cls[0])
            cls_name = plate_detector.names.get(cls_id, "")
            pconf = float(pbox.conf[0])
            px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
            if cls_name in ['plat-nomor', 'license_plate'] or len(plate_detector.names) == 1:
                global_plates.append({
                    "box": [max(0, px1), max(0, py1), min(iw, px2), min(ih, py2)],
                    "conf": pconf,
                    "matched": False
                })
            elif cls_name in ['Hatchback', 'Sedan', 'Sports Utility Vehicle', 'Van', 'Small Bus',
                              'Large Bus', 'Medium Goods Vehicle', 'Pickup Truck', 'Light Goods Vehicle']:
                body_hints.append({
                    "name": cls_name,
                    "conf": pconf,
                    "box": [max(0, px1), max(0, py1), min(iw, px2), min(ih, py2)]
                })

        # Deteksi Fallback untuk Plat Hitam / Gelap / Plat Militer
        if not global_plates or max([p["conf"] for p in global_plates], default=0.0) < 0.30:
            pdet_legacy_full = plate_model_legacy.predict(img, conf=0.10, device=DEVICE, verbose=False)[0]
            for pbox in pdet_legacy_full.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                pconf = float(pbox.conf[0])
                box = [max(0, px1), max(0, py1), min(iw, px2), min(ih, py2)]
                if is_valid_plate_box(box, iw, ih):
                    if not any(compute_iou(box, ep["box"]) > 0.35 for ep in global_plates):
                        global_plates.append({
                            "box": box,
                            "conf": pconf,
                            "matched": False
                        })

        # FALLBACK UNTUK MOBIL GELAP / HITAM (hanya jika sudah ada kendaraan di ROI atau mode foto)
        if not candidates and global_plates:
            best_p = max(global_plates, key=lambda p: p["conf"])
            if best_p["conf"] >= 0.25:
                px1, py1, px2, py2 = best_p["box"]
                found_box = None
                for box in vdet.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    if bx1 - 30 <= (px1 + px2) / 2 <= bx2 + 30 and by1 - 30 <= (py1 + py2) / 2 <= by2 + 30:
                        found_box = box
                        break
                if found_box is not None:
                    v_type = map_vehicle_class_name(found_box.cls[0], vehicle_model)
                    v_conf = float(found_box.conf[0])
                    x1, y1, x2, y2 = map(int, found_box.xyxy[0].tolist())
                    inside, overlap = is_vehicle_inside_roi(x1, y1, x2, y2, iw, ih)
                    if inside or not is_stream:
                        candidates.append({
                            "track_id": None,
                            "vehicle_type": v_type,
                            "v_conf": v_conf,
                            "x1": max(0, x1), "y1": max(0, y1),
                            "x2": min(iw, x2), "y2": min(ih, y2),
                            "area": (x2 - x1) * (y2 - y1),
                            "_inside_roi": inside,
                            "_roi_overlap": overlap,
                            "front_score": 1.0
                        })

    t_plate_total += (time.time() - t_plate_block0) if plate_stage_ran else 0.0
    t_yolo = time.time() - t_yolo_0

    results_out = []
    t_body_total = 0.0
    t_ocr_total = 0.0

    if candidates:
        for cand in candidates:
            initial_vtype = cand["vehicle_type"]
            v_conf = cand["v_conf"]
            cand_track_id = cand.get("track_id")
            x1, y1, x2, y2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
            vehicle_crop = img[y1:y2, x1:x2]

            # Cek status ROI / Detection Area untuk kendaraan ini
            is_in_roi, roi_overlap = is_vehicle_inside_roi(x1, y1, x2, y2, iw, ih)

            tid, existing_trk = confirmation_manager.get_or_create_track(
                cand_track_id, [x1, y1, x2, y2], img_w=iw, img_h=ih
            )
            cand_track_id = tid
            emit_track_event_once(existing_trk, "vehicle_detected", vehicle_type=initial_vtype, bbox=[x1, y1, x2, y2])
            if is_in_roi:
                if existing_trk.roi_entry_time is None:
                    existing_trk.roi_entry_time = time.time()
                emit_track_event_once(existing_trk, "entered_roi")

            # Cek apakah ada deteksi 'bus' di area kendaraan ini
            has_bus_det = any(
                map_vehicle_class_name(b.cls[0], vehicle_model) == 'bus' and
                compute_iou([x1, y1, x2, y2], list(map(int, b.xyxy[0].tolist()))) > 0.30
                for b in vdet.boxes
            )

            # 3. BODY STYLE CACHING PER TRACK ID
            t_b0 = time.time()
            existing_trk = confirmation_manager.tracks.get(cand_track_id) if cand_track_id else None
            if existing_trk and existing_trk.cached_body_style and existing_trk.cached_body_conf >= 0.80:
                # REUSE CACHED RESULT (0 ms!)
                vehicle_type = existing_trk.cached_vtype or initial_vtype
                body_style = existing_trk.cached_body_style
                body_style_conf = existing_trk.cached_body_conf
            else:
                vehicle_type, body_style, body_style_conf = classify_vehicle_indonesian(
                    img, [x1, y1, x2, y2], initial_vtype, v_conf, has_bus_det=has_bus_det, body_hints=body_hints
                )
                if existing_trk:
                    existing_trk.cached_body_style = body_style
                    existing_trk.cached_body_conf = body_style_conf
                    existing_trk.cached_vtype = vehicle_type
            t_body_total += (time.time() - t_b0)

            # JIKA DI LUAR DETECTION AREA ATAU LOST INTEREST:
            # Tidak memicu deteksi plat nomor atau OCR!
            if (not is_in_roi) or existing_trk.lost_interest:
                results_out.append({
                    "track_id": cand_track_id,
                    "vehicle_type": vehicle_type,
                    "body_style": body_style,
                    "body_style_confidence": round(body_style_conf, 3) if body_style_conf else None,
                    "license_plate": None,
                    "ocr_method": None,
                    "vehicle_confidence": round(v_conf, 3),
                    "plate_confidence": None,
                    "plate_bbox_confidence": None,
                    "ocr_confidence": None,
                    "bbox": [x1, y1, x2, y2],
                    "plate_bbox": None,
                    "plate_crop": None,
                    "image_width": iw,
                    "image_height": ih,
                    "inside_interest_area": bool(is_in_roi),
                    "plate_detected": False,
                    "lost_interest": bool(existing_trk.lost_interest)
                })
                continue

            # JIKA DI DALAM DETECTION AREA (INSIDE ROI):
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

            if matched_plate is not None and matched_plate.get("conf", 0.0) >= 0.12:
                gpx1, gpy1, gpx2, gpy2 = matched_plate["box"]
                if is_valid_plate_box([gpx1, gpy1, gpx2, gpy2], iw, ih):
                    plate_crop = crop_plate_with_padding(img, gpx1, gpy1, gpx2, gpy2)
                    plate_conf_val = matched_plate["conf"]
                    abs_plate_bbox = [gpx1, gpy1, gpx2, gpy2]

            # Hierarchical Plate Crop (jika belum terdeteksi dari full frame)
            if plate_crop.size == 0 and vehicle_crop.size > 0:
                _tp = time.time()
                pdet_crop = plate_detector.predict(vehicle_crop, conf=0.10, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
                t_plate_total += time.time() - _tp
                valid_crops = []
                for b in pdet_crop.boxes:
                    cls_id = int(b.cls[0])
                    cname = plate_detector.names.get(cls_id, "")
                    if cname not in ['plat-nomor', 'license_plate'] and len(plate_detector.names) > 1:
                        continue
                    cpx1, cpy1, cpx2, cpy2 = map(int, b.xyxy[0].tolist())
                    abs_box = [x1 + cpx1, y1 + cpy1, x1 + cpx2, y1 + cpy2]
                    if is_valid_plate_box(abs_box, iw, ih):
                        valid_crops.append((abs_box, float(b.conf[0])))
                if not valid_crops or max([c[1] for c in valid_crops], default=0.0) < 0.30:
                    _tp = time.time()
                    pdet_legacy = plate_model_legacy.predict(vehicle_crop, conf=min(p_conf_thresh, 0.06), device=DEVICE, verbose=False)[0]
                    t_plate_total += time.time() - _tp
                    for b in pdet_legacy.boxes:
                        cpx1, cpy1, cpx2, cpy2 = map(int, b.xyxy[0].tolist())
                        abs_box = [x1 + cpx1, y1 + cpy1, x1 + cpx2, y1 + cpy2]
                        if is_valid_plate_box(abs_box, iw, ih):
                            valid_crops.append((abs_box, float(b.conf[0])))
                if valid_crops:
                    valid_crops.sort(key=lambda x: -x[1])
                    best_abs, best_c = valid_crops[0]
                    plate_conf_val = best_c
                    abs_plate_bbox = best_abs
                    plate_crop = crop_plate_with_padding(img, abs_plate_bbox[0], abs_plate_bbox[1],
                                                         abs_plate_bbox[2], abs_plate_bbox[3])
                elif matched_plate is not None:
                    gpx1, gpy1, gpx2, gpy2 = matched_plate["box"]
                    if is_valid_plate_box([gpx1, gpy1, gpx2, gpy2], iw, ih):
                        plate_crop = crop_plate_with_padding(img, gpx1, gpy1, gpx2, gpy2)
                        plate_conf_val = matched_plate["conf"]
                        abs_plate_bbox = [gpx1, gpy1, gpx2, gpy2]
            elif matched_plate is not None and plate_crop.size == 0:
                gpx1, gpy1, gpx2, gpy2 = matched_plate["box"]
                if is_valid_plate_box([gpx1, gpy1, gpx2, gpy2], iw, ih):
                    plate_crop = crop_plate_with_padding(img, gpx1, gpy1, gpx2, gpy2)
                    plate_conf_val = matched_plate["conf"]
                    abs_plate_bbox = [gpx1, gpy1, gpx2, gpy2]

            if abs_plate_bbox is not None:
                if existing_trk.first_plate_time is None:
                    existing_trk.first_plate_time = time.time()
                emit_track_event_once(existing_trk, "plate_detected", plate_bbox=abs_plate_bbox)

            plate_text = None
            ocr_method = None
            ocr_conf = 0.0

            # 4. PLATE READING (ASYNCHRONOUS DI STREAM, SYNCHRONOUS DI PHOTO UPLOAD)
            if plate_crop.size > 0:
                crop_q = compute_crop_quality(plate_crop, plate_conf_val)

                if is_stream:
                    # Jalur Stream: Asynchronous OCR menggunakan crop terbaik
                    if existing_trk.is_locked or existing_trk.history_saved:
                        plate_text = existing_trk.confirmed_data.get("license_plate") if existing_trk.confirmed_data else existing_trk.best_candidate_display
                        ocr_conf = existing_trk.confirmed_data.get("plate_confidence", 0.0) if existing_trk.confirmed_data else existing_trk.best_candidate_conf
                        ocr_method = "locked_confirmed"
                    elif not existing_trk.lost_interest:
                        should_run_async_ocr = False
                        if not existing_trk.ocr_pending:
                            if (existing_trk.best_candidate is None) or (crop_q > existing_trk.best_crop_quality * 1.15) or (time.time() - (existing_trk.last_ocr_time or 0) > 0.40):
                                should_run_async_ocr = True
                        elif crop_q > existing_trk.best_crop_quality * 1.25:
                            should_run_async_ocr = True

                        if crop_q <= 0.0:
                            should_run_async_ocr = False   # crop terlalu kecil -> jangan buang waktu OCR

                        if should_run_async_ocr:
                            job_id = get_next_ocr_job_id()
                            existing_trk.ocr_pending = True
                            existing_trk.last_ocr_job_id = job_id
                            existing_trk.best_crop_quality = max(existing_trk.best_crop_quality, crop_q)
                            ph, pw = plate_crop.shape[:2]
                            print(f"[PLATE] track={cand_track_id} frame={frame_id} job={job_id} bbox={abs_plate_bbox} "
                                  f"crop={pw}x{ph} crop_quality={crop_q:.2f} plate_det_conf={plate_conf_val}")
                            emit_track_event_once(existing_trk, "ocr_analyzing", job_id=job_id)
                            ocr_executor.submit(_async_ocr_task, job_id, cand_track_id, plate_crop.copy(), crop_q, time.time(),
                                                {"frame_id": frame_id, "capture_ts": capture_ts, "plate_bbox": abs_plate_bbox})

                        if existing_trk.best_candidate_display:
                            plate_text = existing_trk.best_candidate_display
                            ocr_conf = existing_trk.best_candidate_conf
                            ocr_method = "temporal_best_candidate"
                        else:
                            # Coba pass cepat char_model (~35ms) agar hasil awal tampil seketika
                            t_o0 = time.time()
                            c_raw, c_conf, _ = read_plate_with_char_model(plate_crop, conf=0.08)
                            t_ocr_total += (time.time() - t_o0)
                            prev_norm = normalize_indonesian_plate(c_raw)
                            if prev_norm["valid"] and c_conf >= 0.50:
                                # Preview cepat untuk Latest Detection; BUKAN evidence konfirmasi (lihat NON_EVIDENCE_OCR_METHODS)
                                plate_text = prev_norm["text"]
                                ocr_conf = c_conf
                                ocr_method = "char_model_preview"
                else:
                    # Jalur Single Photo: Synchronous OCR agar hasil lengkap seketika di respons JSON
                    t_o0 = time.time()
                    ensemble_res = ensemble_plate_reading(plate_crop)
                    plate_text = ensemble_res.get("final")
                    ocr_method = ensemble_res.get("method")
                    ocr_conf = float(ensemble_res.get("confidence", 0.0) or 0.0)
                    t_ocr_total += (time.time() - t_o0)

            # Prioritaskan ocr_conf saat plat terbaca agar evaluasi konfirmasi mengukur akurasi OCR
            effective_plate_conf = ocr_conf if (plate_text and ocr_conf > 0) else (plate_conf_val or 0.0)

            results_out.append({
                "track_id": cand_track_id,
                "vehicle_type": vehicle_type,
                "body_style": body_style,
                "body_style_confidence": round(body_style_conf, 3) if body_style_conf else None,
                "license_plate": plate_text,
                "ocr_method": ocr_method,
                "vehicle_confidence": round(v_conf, 3),
                "plate_confidence": round(effective_plate_conf, 3) if effective_plate_conf else None,
                "plate_bbox_confidence": round(plate_conf_val, 3) if plate_conf_val else None,
                "ocr_confidence": round(ocr_conf, 3) if ocr_conf else None,
                "bbox": [x1, y1, x2, y2],
                "plate_bbox": abs_plate_bbox,
                "plate_crop": plate_crop if is_stream else None,
                "image_width": iw,
                "image_height": ih,
                "inside_interest_area": True,
                "plate_detected": bool(abs_plate_bbox is not None)
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
    perf_stats.add("vehicle_detection_ms", t_vehicle * 1000.0)
    if plate_stage_ran:
        perf_stats.add("plate_detection_ms", t_plate_total * 1000.0)
    print(f"[PERF] Vehicle: {round(t_vehicle*1000, 1)}ms | Plate: {round(t_plate_total*1000, 1)}ms | YOLO: {round(t_yolo*1000, 1)}ms | Tracking: {round(t_track*1000, 1)}ms | Body: {round(t_body_total*1000, 1)}ms | OCR: {round(t_ocr_total*1000, 1)}ms | Total: {round(t_total*1000, 1)}ms | FPS: {round(fps, 1)}")

    return {
        "detections": results_out,
        "detection_time_sec": round(t_total, 3),
        "source": image_path,
        "image_width": iw,
        "image_height": ih,
        "perf_breakdown": {
            "yolo_ms": round(t_yolo * 1000, 1),
            "vehicle_ms": round(t_vehicle * 1000, 1),
            "plate_ms": round(t_plate_total * 1000, 1),
            "track_ms": round(t_track * 1000, 1),
            "body_ms": round(t_body_total * 1000, 1),
            "ocr_ms": round(t_ocr_total * 1000, 1),
            "total_ms": round(t_total * 1000, 1),
            "fps": round(fps, 1)
        }
    }



app = Flask(__name__)
sock = Sock(app)


# ============================================================
# REAL-TIME BACKGROUND STREAM INFERENCE WORKER
# Memproses frame camera_stream_manager dan menyiarkan hasil deteksi via WebSocket secara live.
# ============================================================
class StreamInferenceWorker:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.last_payload = None

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True, name="StreamInferenceWorker")
            self.thread.start()
            print("[INFO] StreamInferenceWorker aktif di latar belakang.")

    def stop(self):
        with self.lock:
            self.running = False
        print("[INFO] StreamInferenceWorker dihentikan.")

    def _loop(self):
        last_processed_frame_id = -1
        while self.running:
            with camera_stream_manager.condition:
                if not camera_stream_manager.running:
                    camera_stream_manager.condition.wait(timeout=0.2)
                    continue
                camera_stream_manager.condition.wait_for(
                    lambda: camera_stream_manager.frame_id != last_processed_frame_id or not camera_stream_manager.running,
                    timeout=0.1
                )
                if not self.running or not camera_stream_manager.running:
                    continue
                frame = camera_stream_manager.get_latest_frame()
                frame_id = camera_stream_manager.frame_id
                frame_capture_time = camera_stream_manager.latest_frame_time
                last_processed_frame_id = frame_id

            if frame is None:
                time.sleep(0.02)
                continue

            t0 = time.time()
            try:
                result = run_anpr(frame, is_stream=True, frame_id=frame_id, capture_ts=frame_capture_time)
                det_time_ms = round((time.time() - t0) * 1000)

                # Evaluasi lost tracks (finalize or discard; save_parking_record already broadcasts)
                lost_confirmed, newly_lost = confirmation_manager.check_lost_tracks()
                handle_lost_tracks(newly_lost)
                for trk in lost_confirmed:
                    if trk.confirmed_data and not trk.history_saved:
                        rec = save_parking_record(trk.confirmed_data, source_img=frame)
                        trk.history_saved = True
                        trk.status = "HISTORY_SAVED"

                detections = result.get("detections", [])
                det_ids = {d.get("track_id") for d in detections}

                # Refresh lost_interest flags after check_lost_tracks
                now_ts = time.time()
                with confirmation_manager.lock:
                    for det in detections:
                        trk = confirmation_manager.tracks.get(det.get("track_id"))
                        if trk:
                            det["lost_interest"] = trk.lost_interest
                            det["last_seen"] = round(trk.last_seen, 3)
                            det["last_seen_age_ms"] = int(round(max(0.0, now_ts - trk.last_seen) * 1000))
                            if trk.lost_interest and det.get("status") not in ("DISCARDED", "CONFIRMED", "HISTORY_SAVED"):
                                det["status"] = trk.status if trk.status in ("DISCARDED", "CONFIRMED", "HISTORY_SAVED") else "LOST_INTEREST"
                                det["ocr_status"] = "LOST_INTEREST" if det["status"] == "LOST_INTEREST" else det["status"]

                    for trk in newly_lost:
                        if trk.track_id not in det_ids:
                            detections.append(trk.debug_snapshot(now_ts))

                # Check newly confirmed (save_parking_record broadcasts entry_confirmed once)
                for det in detections:
                    if det.get("is_newly_confirmed"):
                        rec = save_parking_record(det, source_img=frame)
                        det["record"] = rec

                # Broadcast detection update ke seluruh WebSocket client
                now_broadcast = time.time()
                frame_age_ms = int(round((now_broadcast - frame_capture_time) * 1000)) if frame_capture_time else det_time_ms
                perf_stats.add("frame_age_ms", frame_age_ms)
                payload = {
                    "type": "detection_update",
                    "frame_id": frame_id,
                    "timestamp": t0,
                    "capture_timestamp": frame_capture_time,
                    "frame_age_ms": frame_age_ms,
                    "latency_ms": det_time_ms,
                    "alpr_state": compute_alpr_state(detections),
                    "detections": detections,
                    "interest_area": INTEREST_AREA,
                    "image_width": result.get("image_width", 1920),
                    "image_height": result.get("image_height", 1080),
                }
                self.last_payload = payload
                ws_broadcaster.broadcast(payload)
            except Exception as e:
                print(f"[STREAM INFERENCE ERROR]: {e}")
                time.sleep(0.05)


stream_inference_worker = StreamInferenceWorker()


# ============================================================
# WEBSOCKET REAL-TIME STREAMING ENDPOINT
# ============================================================
@sock.route('/ws/live')
def live_ws(ws):
    ws_broadcaster.register(ws)
    try:
        init_payload = {
            "type": "init",
            "interest_area": INTEREST_AREA,
            "status": camera_stream_manager.get_status(),
            "alpr_state": (stream_inference_worker.last_payload or {}).get("alpr_state", "WAITING_FOR_VEHICLE"),
            "history": list(LATEST_RECORDS)[:30]
        }
        ws.send(json.dumps(init_payload))

        while True:
            data = ws.receive(timeout=1.0)
            if data is not None:
                try:
                    msg = json.loads(data)
                    m_type = msg.get("type")
                    if m_type == "ping":
                        ws.send(json.dumps({"type": "pong", "time": time.time()}))
                    elif m_type == "detect_frame":
                        b64_img = msg.get("image", "")
                        if "," in b64_img:
                            b64_img = b64_img.split(",", 1)[1]
                        if b64_img:
                            img_bytes = base64.b64decode(b64_img)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                t0 = time.time()
                                res = run_anpr(frame, is_stream=True)
                                det_ms = round((time.time() - t0) * 1000)

                                lost_confirmed, newly_lost = confirmation_manager.check_lost_tracks()
                                handle_lost_tracks(newly_lost)
                                for trk in lost_confirmed:
                                    if trk.confirmed_data and not trk.history_saved:
                                        rec = save_parking_record(trk.confirmed_data, source_img=frame)
                                        trk.history_saved = True
                                        trk.status = "HISTORY_SAVED"

                                detections = res.get("detections", [])
                                det_ids = {d.get("track_id") for d in detections}
                                now_ts = time.time()
                                with confirmation_manager.lock:
                                    for trk in newly_lost:
                                        if trk.track_id not in det_ids:
                                            detections.append(trk.debug_snapshot(now_ts))

                                for d in detections:
                                    if d.get("is_newly_confirmed"):
                                        rec = save_parking_record(d, source_img=frame)
                                        d["record"] = rec

                                payload = {
                                    "type": "detection_update",
                                    "timestamp": t0,
                                    "latency_ms": det_ms,
                                    "alpr_state": compute_alpr_state(detections),
                                    "detections": detections,
                                    "interest_area": INTEREST_AREA,
                                    "image_width": res.get("image_width", frame.shape[1]),
                                    "image_height": res.get("image_height", frame.shape[0]),
                                }
                                ws.send(json.dumps(payload))
                except Exception as ex:
                    print(f"[WS MSG ERROR]: {ex}")
    except Exception:
        pass
    finally:
        ws_broadcaster.unregister(ws)


@app.route("/api/config/interest_area", methods=["GET", "POST", "OPTIONS"])
def config_interest_area():
    """Mengambil atau memperbarui konfigurasi Detection / Interest Area (ROI)."""
    global INTEREST_AREA
    if request.method == "OPTIONS":
        return "", 200
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        x_min = max(0.0, min(1.0, float(data.get("x_min", INTEREST_AREA["x_min"]))))
        y_min = max(0.0, min(1.0, float(data.get("y_min", INTEREST_AREA["y_min"]))))
        x_max = max(0.0, min(1.0, float(data.get("x_max", INTEREST_AREA["x_max"]))))
        y_max = max(0.0, min(1.0, float(data.get("y_max", INTEREST_AREA["y_max"]))))
        if x_max > x_min and y_max > y_min:
            INTEREST_AREA = {
                "x_min": round(x_min, 3),
                "y_min": round(y_min, 3),
                "x_max": round(x_max, 3),
                "y_max": round(y_max, 3)
            }
            ws_broadcaster.broadcast({
                "type": "interest_area_update",
                "interest_area": INTEREST_AREA
            })
            return jsonify({"status": "ok", "interest_area": INTEREST_AREA})
        else:
            return jsonify({"error": "Invalid ROI bounds: x_max must be > x_min and y_max > y_min"}), 400
    return jsonify({"status": "ok", "interest_area": INTEREST_AREA})


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
        "stream_status": camera_stream_manager.get_status(),
        "interest_area": INTEREST_AREA
    })



@app.route("/api/perf", methods=["GET"])
def perf():
    """Ringkasan latency & jumlah panggilan OCR (rolling). Dipakai untuk benchmark PaddleOCR."""
    return jsonify({
        "ocr_engine": {"name": "paddleocr", "version": paddle_engine.version, "api": paddle_engine.api,
                       "device": PADDLE_DEVICE, "init_ms": round(paddle_engine.init_ms), "warmup_ms": round(paddle_engine.warmup_ms)},
        "summary": perf_stats.summary(),
        "vehicles": list(perf_stats.completed)[-50:],
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
    Request-response snapshot of the latest detection state.
    If StreamInferenceWorker is already running, return the last WebSocket payload
    instead of running a duplicate ANPR pass.
    """
    if request.method == "OPTIONS":
        return "", 200

    if stream_inference_worker.running and stream_inference_worker.last_payload:
        payload = dict(stream_inference_worker.last_payload)
        payload.pop("type", None)
        return jsonify(payload)

    frame = camera_stream_manager.get_latest_frame()
    if frame is None:
        return jsonify({"error": "Belum ada frame video di memory. Pastikan kamera CCTV sudah terhubung dan aktif."}), 400

    t0 = time.time()
    result = run_anpr(frame, is_stream=True)
    det_time = time.time() - t0
    result["detection_time_sec"] = round(det_time, 3)
    result["interest_area"] = INTEREST_AREA

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
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Gagal mendekode file gambar"}), 400

    try:
        t0 = time.time()
        is_stream = request.form.get("is_stream", "false").lower() in ["true", "1"]
        result = run_anpr(img, is_stream=is_stream)
        det_time = time.time() - t0
        result["detection_time_sec"] = round(det_time, 3)
        # Simpan ke memory RAM jika kendaraan terkonfirmasi (atau single photo upload)
        if result.get("detections"):
            primary_det = result["detections"][0]
            primary_det["latency_ms"] = round(det_time * 1000)
            if primary_det.get("is_newly_confirmed"):
                rec = save_parking_record(primary_det, source_img=img)
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
    3. Menjalankan model AI ANPR lokal (YOLO + PaddleOCR).
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
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)