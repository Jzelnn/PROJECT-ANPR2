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
from flask_cors import CORS
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

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from ultralytics import YOLO
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")

import paddle
from paddle import inference as paddle_inference
from paddlex.inference.models.runners.paddle_static import runner as static_runner

# Patch PaddleStaticRunner on CPU to disable oneDNN PIR attribution bug in PaddlePaddle 3.3.1
orig_static_create = static_runner.PaddleStaticRunner._create
def _custom_static_create(self):
    model_paths = static_runner.get_model_paths(self.model_dir, self.model_file_prefix)
    model_file, params_file = model_paths['paddle']
    config = paddle_inference.Config(str(model_file), str(params_file))
    if self._config.get("device_type") != "cpu" and HAS_CUDA:
        config.enable_use_gpu(100, self._config.get("device_id", 0))
    else:
        config.disable_gpu()
        if hasattr(config, "disable_onednn"):
            config.disable_onednn()
        if hasattr(config, "disable_mkldnn"):
            config.disable_mkldnn()
        config.set_cpu_math_library_num_threads(4)
        if hasattr(config, "enable_new_ir"):
            config.enable_new_ir(False)
    config.disable_glog_info()
    return paddle_inference.create_predictor(config)

static_runner.PaddleStaticRunner._create = _custom_static_create

import paddleocr
from paddleocr import PaddleOCR

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
import config
CAPTURES_DIR = config.SCREENSHOTS_DIR
os.makedirs(CAPTURES_DIR, exist_ok=True)

# ============================================================
# PERFORMANCE & DETECTION CONFIGURATION
# Configurable parameters for speed, detection, and temporal confirmation
# ============================================================
IMG_SIZE = int(os.environ.get("ANPR_IMG_SIZE", 640))           # 640 for reliable small plate detection
VEHICLE_CONF_THRESH = float(os.environ.get("ANPR_VEHICLE_CONF", 0.38))
MOTORCYCLE_CONF_THRESH = float(os.environ.get("ANPR_MOTOR_CONF", 0.25))
PLATE_CONF_THRESH = float(os.environ.get("ANPR_PLATE_CONF", 0.20))
IOU_THRESH = float(os.environ.get("ANPR_IOU_THRESH", 0.35))
FRAME_SKIP = int(os.environ.get("ANPR_FRAME_SKIP", 1))         # Process 1 of every N frames (1 = all, 2 = half)
DEVICE = os.environ.get("ANPR_DEVICE", "cuda" if HAS_CUDA else "cpu")
USE_FP16 = False  # Keep false on CPU to prevent warnings

# Fast Temporal Confirmation Configuration
MIN_OBSERVATIONS = 3
MAX_OBSERVATIONS = 5
TEMPORAL_WINDOW_SEC = 0.50
CONSISTENCY_THRESH = 0.70
CONFIRM_CONF_THRESH = 0.80

# ============================================================
# DETECTION / INTEREST AREA CONFIGURATION (ROI)
# Kendaraan hanya menjadi target aktif ALPR setelah memasuki area ini.
# Bounding box normalisasi [x_min, y_min, x_max, y_max] (0.0 s/d 1.0).
# ============================================================
# ============================================================
# DETECTION / INTEREST AREA CONFIGURATION (ROI)
# Kendaraan hanya menjadi target aktif ALPR setelah memasuki area ini.
# Bounding box tegak lurus (non-slanted rectangle) [x_min, y_min, x_max, y_max] (0.0 s/d 1.0).
# ============================================================
DEFAULT_INTEREST_AREA = {
    "x_min": float(os.environ.get("ANPR_ROI_XMIN", 0.265)),
    "y_min": float(os.environ.get("ANPR_ROI_YMIN", 0.20)),
    "x_max": float(os.environ.get("ANPR_ROI_XMAX", 0.60)),
    "y_max": float(os.environ.get("ANPR_ROI_YMAX", 0.71)),
    "points": [
        [0.265, 0.20],
        [0.60, 0.20],
        [0.60, 0.71],
        [0.265, 0.71]
    ]
}
INTEREST_AREA = dict(DEFAULT_INTEREST_AREA)

# Lost Interest parameters
LOST_INTEREST_TIMEOUT_SEC = float(os.environ.get("ANPR_LOST_TIMEOUT", 2.0))
LOST_INTEREST_OUTSIDE_FRAMES = int(os.environ.get("ANPR_LOST_FRAMES", 4))
# Plate detection refresh: do not queue duplicate/stale plate inference.
# Re-run only when the cached plate is old enough or the previous attempt failed.
PLATE_REDETECT_INTERVAL_SEC = float(os.environ.get("ANPR_PLATE_REDETECT_SEC", 0.35))
PLATE_INFER_IMGSZ = int(os.environ.get("ANPR_PLATE_IMGSZ", 640))
PLATE_INFER_CONF = float(os.environ.get("ANPR_PLATE_INFER_CONF", 0.15))



def is_vehicle_inside_roi(x1, y1, x2, y2, img_w, img_h, roi=None):
    """
    Mengecek apakah kendaraan masuk AREA DETEKTOR (ROI).
    Responsif seketika: Begitu bagian kendaraan (center, bumper, atau >= 15% bodi) masuk area deteksi,
    kendaraan langsung aktif. Begitu keluar dari area deteksi, langsung nonaktif.
    """
    if roi is None:
        roi = INTEREST_AREA

    rx1 = roi["x_min"] * img_w
    ry1 = roi["y_min"] * img_h
    rx2 = roi["x_max"] * img_w
    ry2 = roi["y_max"] * img_h

    # 1. Cek perpotongan bounding box dengan area deteksi
    ix1 = max(float(x1), rx1)
    iy1 = max(float(y1), ry1)
    ix2 = min(float(x2), rx2)
    iy2 = min(float(y2), ry2)

    if ix2 > ix1 and iy2 > iy1:
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        veh_area = max(1.0, float((x2 - x1) * (y2 - y1)))
        overlap_ratio = inter_area / veh_area
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        center_in = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

        # Aktif seketika jika center masuk atau minimal 15% bodi menyentuh area deteksi
        if center_in or overlap_ratio >= 0.15:
            return True, overlap_ratio

    return False, 0.0


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

    def _async_write_snapshot(dst, img_data, img_path):
        try:
            if img_data is not None:
                cv2.imwrite(dst, img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            elif img_path and os.path.exists(img_path):
                shutil.copyfile(img_path, dst)
        except Exception as e:
            print(f"[WARN] Async snapshot save error: {e}")

    if source_img is not None:
        ai_pool.submit(_async_write_snapshot, dest_path, source_img.copy(), None)
    elif source_img_path and os.path.exists(source_img_path):
        ai_pool.submit(_async_write_snapshot, dest_path, None, source_img_path)

    timing_data = det.get("timing") or {}
    proc_ms = det.get("processing_ms") or det.get("latency_ms")
    if proc_ms is None and timing_data.get("total_ms"):
        proc_ms = int(round(timing_data["total_ms"]))

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
        "processing_ms": proc_ms,
        "timing": timing_data,
        "snapshot_url": f"/captures/{snapshot_filename}"
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
    print(f"[PERF_LATENCY] [HISTORY_SAVE: track_id={track_id} plate='{safe_plate}' time={now_epoch:.3f} time_from_confirmation_ms={dt_conf_to_hist:.1f}]", flush=True)
    return rec



# ============================================================
# PERSISTENT CAMERA STREAM MANAGER (BACKGROUND RTSP / MJPEG WORKER)
# Menjaga koneksi RTSP tetap hidup di latar belakang agar:
# 1. Live stream di monitor CCTV benar-benar bergerak mulus (25 FPS).
# 2. Deteksi instan: frame selalu siap di RAM sehingga scan < 1 detik (bebas jeda RTSP 3s).
# ============================================================
class CameraStreamManager:
    def __init__(self):
        self.lock = threading.RLock()
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
            worker = globals().get('stream_inference_worker')
            if worker:
                worker.start()
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
        old_thread = self.thread
        self.thread = None
        worker = globals().get('stream_inference_worker')
        if worker:
            worker.stop()
        if 'confirmation_manager' in globals():
            confirmation_manager.reset()
        self.condition.notify_all()
        if old_thread and old_thread.is_alive() and old_thread != threading.current_thread():
            old_thread.join(timeout=0.5)


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
            # Resolve relative file path to PROJECT_ROOT
            if not os.path.isabs(url):
                cand_path = os.path.join(config.PROJECT_ROOT, url)
                if os.path.exists(cand_path):
                    url = cand_path
            # Anggap IP camera RTSP jika bukan file
            if not os.path.isfile(url):
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
            if not self.running:
                self.status = "disconnected"
        print("[STREAM] Koneksi video telah ditutup dengan aman.")


camera_stream_manager = CameraStreamManager()

print("[INFO] Memuat model AI...")
# 1. Model Deteksi Kendaraan Indonesia (dilatih khusus untuk kendaraan jalanan Indonesia)
indo_vmodel_path = config.VEHICLE_MODEL_PATH
if os.path.exists(indo_vmodel_path):
    vehicle_model = YOLO(indo_vmodel_path)
    print(f"[INFO] Model Kendaraan Indonesia aktif: {os.path.basename(indo_vmodel_path)} ({len(vehicle_model.names)} kelas)")
else:
    fallback_vmodel = os.path.join(config.MODELS_DIR, "_archive", "vehicle_old_4class.pt")
    vehicle_model = YOLO(fallback_vmodel)
    print(f"[INFO] Model Kendaraan Standar aktif: {os.path.basename(fallback_vmodel)} ({len(vehicle_model.names)} kelas)")

# 2. Model Deteksi Plat Nomor & Sub-Tipe Bodi (PRKING-ANPR-1 Best Checkpoint)
best_plate_path = config.PLATE_MODEL_PATH
if os.path.exists(best_plate_path):
    plate_detector = YOLO(best_plate_path)
    print(f"[INFO] Model Plat Nomor & Sub-Tipe Bodi Best aktif: {os.path.basename(best_plate_path)} ({len(plate_detector.names)} kelas)")
else:
    plate_detector = YOLO(config.PLATE_LEGACY_MODEL_PATH)
    print(f"[INFO] Model Plat Nomor Standar aktif: {os.path.basename(config.PLATE_LEGACY_MODEL_PATH)}")

plate_model_legacy = YOLO(config.PLATE_LEGACY_MODEL_PATH)
body_style_model = YOLO(config.BODY_TYPE_MODEL_PATH)
char_model = YOLO(config.CHARACTER_MODEL_PATH)

class PaddleEngine:
    """
    Wrapper PaddleOCR tunggal. Diinisialisasi SEKALI saat startup.
    Thread-safe menggunakan lock internal.
    """
    def __init__(self, device="cpu"):
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
            self.engine = PaddleOCR(**kw)
        else:
            self.engine = PaddleOCR(lang="en", use_angle_cls=False, use_gpu=(device != "cpu"), show_log=False)
        self.init_ms = (time.time() - t0) * 1000.0
        print(f"[PADDLE] siap | paddleocr={self.version} api={self.api} device={device} init={self.init_ms:.0f}ms")

    def run(self, img_bgr):
        with self.lock:
            if self.api == "v3":
                out = self.engine.predict(img_bgr)
            else:
                out = self.engine.ocr(img_bgr, cls=False)
        items = []
        if self.api == "v3":
            for res in (out or []):
                d = res if isinstance(res, dict) else (getattr(res, "json", None) or {})
                if isinstance(d.get("res"), dict):
                    d = d["res"]
                texts = d.get("rec_texts") or []
                scores = d.get("rec_scores") or []
                for i, t in enumerate(texts):
                    sc = float(scores[i]) if i < len(scores) else 0.0
                    items.append({"text": str(t), "conf": sc})
        else:
            page = out[0] if out else None
            for line in (page or []):
                try:
                    box, (t, sc) = line
                    items.append({"text": str(t), "conf": float(sc)})
                except Exception:
                    pass
        return items

paddle_engine = PaddleEngine(device="cpu")
ai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR_YOLO")
plate_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR_ASYNC_PLATE")
ocr_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ANPR_ASYNC_OCR")
body_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR_ASYNC_BODY")
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


def is_tax_date(text):
    """Mendeteksi tanggal pajak / bulan-tahun pada plat nomor (e.g. '09-25', '02.28', '11.44', '1328')"""
    if not text:
        return False
    raw = str(text).strip()
    clean = re.sub(r'[^0-9\.\-]', '', raw)
    if re.match(r'^(0[1-9]|1[0-2])[\.\-](2[0-9]|3[0-9])$', clean):
        return True
    if len(clean) == 4 and clean.isdigit():
        mm = int(clean[:2])
        yy = int(clean[2:])
        if 1 <= mm <= 12 and 20 <= yy <= 39:
            return True
    if re.match(r'^[A-Z0-9]{2}[\.\-][0-9]{2}$', raw.upper()):
        return True
    return False


def is_valid_indonesian_plate_structure(plate_str):
    """
    Memvalidasi apakah string OCR memenuhi struktur plat nomor Indonesia atau format dinas/militer.
    Returns: (is_valid: bool, plate_type: str or None)
    """
    if not plate_str or plate_str in ["TIDAK_TERBACA", "UNKNOWN", ""]:
        return False, None
    plate_str = str(plate_str).strip()
    if is_tax_date(plate_str):
        return False, None

    # Format dinas / militer: e.g. "523-07", "1234-01", "12345-00" (min 3 digit angka sebelum strip)
    if '-' in plate_str:
        parts = plate_str.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and 3 <= len(parts[0]) <= 5 and len(parts[1]) == 2:
            return True, "military"
        return False, None

    clean = re.sub(r'[^A-Z0-9]', '', plate_str.upper())
    # Plat standar sipil Indonesia minimal 4 karakter (e.g. B 1 A, B 12 AB, B 1234 ABC)
    if len(clean) < 4:
        return False, None

    # Format sipil Indonesia: 1-2 huruf kode wilayah (wajib terdaftar di Samsat), 1-4 angka nomor polisi, 1-3 huruf seri akhir
    m = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', clean)
    if m:
        prefix, digits, suffix = m.groups()
        if prefix in SAMSAT_PREFIXES:
            return True, "standard"
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


def read_plate_with_char_model(plate_crop, conf=0.20, iou_threshold=0.35):
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


def read_plate_with_paddleocr(plate_crop):
    """Membaca teks plat nomor menggunakan PaddleOCR dengan penanganan multi-token & eliminasi tanggal pajak."""
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0, []

    if h < 45:
        scale = 60.0 / h
        proc_crop = cv2.resize(plate_crop, (int(w * scale), 60), interpolation=cv2.INTER_LINEAR)
    elif h > 140:
        scale = 90.0 / h
        proc_crop = cv2.resize(plate_crop, (int(w * scale), 90), interpolation=cv2.INTER_AREA)
    else:
        proc_crop = plate_crop

    items = paddle_engine.run(proc_crop)
    if not items:
        gray = cv2.cvtColor(proc_crop, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < 90.0:
            inv = cv2.bitwise_not(proc_crop)
            items = paddle_engine.run(inv)

    non_date_items = []
    for item in items:
        text = str(item.get("text", "")).strip()
        conf = float(item.get("conf", 0.0) or 0.0)
        if not text or conf < 0.20:
            continue
        if is_tax_date(text):
            continue
        non_date_items.append((text, conf))

    if not non_date_items:
        return "", 0.0, []

    all_raw_texts = [t for t, _ in non_date_items]
    candidates = []

    for t, c in non_date_items:
        candidates.append((t, c))

    if len(non_date_items) > 1:
        joined = " ".join([t for t, _ in non_date_items])
        avg_c = float(np.mean([c for _, c in non_date_items]))
        candidates.append((joined, avg_c))

    valid_cand = []
    for t, c in candidates:
        parsed = refine_indonesian_plate("", t, all_raw_texts)
        if parsed and is_valid_indonesian_plate_structure(parsed)[0]:
            valid_cand.append((parsed, c))

    if valid_cand:
        valid_cand.sort(key=lambda x: -x[1])
        return valid_cand[0][0], valid_cand[0][1], all_raw_texts

    non_date_items.sort(key=lambda x: -x[1])
    return non_date_items[0][0], non_date_items[0][1], all_raw_texts


def refine_indonesian_plate(char_raw, paddle_raw="", all_paddle_texts=None):
    """
    Menyelaraskan hasil pembacaan plat nomor sesuai regulasi Korlantas Polri Indonesia:
    1. Kode Wilayah (Prefix): 1-2 Huruf (dicocokkan dengan SAMSAT_PREFIXES)
    2. Nomor Polisi (Digits): 1-4 Angka
    3. Seri Akhir (Suffix): 1-3 Huruf (Tanpa huruf 'Q')
    Disambiguasi akurat tanpa halusinasi dari string numerik murni.
    """
    if all_paddle_texts is None:
        all_paddle_texts = []

    c_raw = str(char_raw).strip().upper() if char_raw else ""
    p_raw = str(paddle_raw).strip().upper() if paddle_raw else ""

    # 0. DETEKSI PLAT MILITER / TNI / DINAS (Format 3-5 Digit + '-' + 2 Digit, misal 523-07, 151-12)
    for t in ([p_raw, c_raw] + [str(x).strip().upper() for x in all_paddle_texts]):
        if '-' in t:
            parts = t.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and 3 <= len(parts[0]) <= 5 and len(parts[1]) == 2:
                return f"{parts[0]}-{parts[1]}"

    pref_map = {'8': 'B', '0': 'D', '1': 'I', '5': 'S', '2': 'Z', '3': 'E', '4': 'A'}
    suff_map = {'8': 'B', '0': 'O', '1': 'I', '5': 'S', '2': 'Z', '3': 'E', '4': 'A', '6': 'G'}
    digit_map = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0'}

    def _parse_candidate_text(raw_text):
        if not raw_text or is_tax_date(raw_text):
            return ""
        raw = str(raw_text).strip().upper()

        # A. 3 Token Terpisah (misal "B" "2175" "BJH" atau "B" "2049" "88K")
        tokens = [t for t in re.sub(r'[^A-Z0-9]', ' ', raw).split() if t]
        if len(tokens) == 3:
            p_raw, d_raw, s_raw = tokens[0], tokens[1], tokens[2]
            p_tok = ''.join([ch if ch.isalpha() else pref_map.get(ch, ch) for ch in p_raw])
            d_tok = ''.join([ch if ch.isdigit() else digit_map.get(ch, '') for ch in d_raw])
            if any(ch.isalpha() for ch in s_raw):
                s_tok = ''.join([suff_map.get(ch, ch) if ch.isdigit() else ch for ch in s_raw])
                s_tok = re.sub(r'[^A-Z]', '', s_tok)
                if p_tok in SAMSAT_PREFIXES and 1 <= len(d_tok) <= 4 and d_tok.isdigit() and 1 <= len(s_tok) <= 3 and s_tok.isalpha():
                    return f"{p_tok} {d_tok} {s_tok}"

        # B. Teks Gabungan (misal "B2175BJH", "82489PZH", "W6035DK", "B9301TBD")
        clean = re.sub(r'[^A-Z0-9]', '', raw)
        if not clean or len(clean) < 4:
            return ""

        if clean[0] == '8' and len(clean) >= 5 and (clean[1].isdigit() or clean[1] == 'B'):
            clean = 'B' + clean[1:]

        for p_len in [2, 1]:
            if len(clean) >= p_len + 2:
                p_cand = clean[:p_len]
                p_cand_clean = ''.join([ch if ch.isalpha() else pref_map.get(ch, ch) for ch in p_cand])
                if p_cand_clean in SAMSAT_PREFIXES:
                    rem = clean[p_len:]
                    m = re.match(r'^(\d{1,4})([A-Z0-9]{1,3})$', rem)
                    if m:
                        d_cand = m.group(1)
                        s_raw = m.group(2)
                        if any(ch.isalpha() for ch in s_raw):
                            s_cand = ''.join([suff_map.get(ch, ch) if ch.isdigit() else ch for ch in s_raw])
                            s_cand = re.sub(r'[^A-Z]', '', s_cand)
                            if 1 <= len(s_cand) <= 3 and s_cand.isalpha():
                                return f"{p_cand_clean} {d_cand} {s_cand}"

        return ""

    candidates_to_try = []
    if p_raw and not is_tax_date(p_raw):
        candidates_to_try.append(p_raw)
    for ext in all_paddle_texts:
        if ext and ext not in candidates_to_try and not is_tax_date(ext):
            candidates_to_try.append(ext)
    if c_raw and c_raw not in candidates_to_try and not is_tax_date(c_raw):
        candidates_to_try.append(c_raw)

    for cand in candidates_to_try:
        parsed = _parse_candidate_text(cand)
        if parsed:
            is_valid, _ = is_valid_indonesian_plate_structure(parsed)
            if is_valid:
                return parsed

    return ""


def ensemble_plate_reading(plate_crop):
    """
    Menggabungkan hasil deteksi Character Model dan PaddleOCR
    dengan auto-deskewing adaptif dan motion-blur sharpening secara evidence-based.
    """
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0 or plate_crop.shape[0] < 8 or plate_crop.shape[1] < 8:
        return {"final": "", "confidence": 0.0, "method": "invalid_crop"}

    # 0. Enhancement untuk citra plat bergerak (mengurangi motion blur)
    plate_crop = enhance_moving_plate_crop(plate_crop)

    # 1. Pembacaan via PaddleOCR (Akurasi tertinggi)
    paddle_raw, paddle_conf, all_paddle = read_plate_with_paddleocr(plate_crop)
    paddle_raw = paddle_raw.upper()

    # 2. Pembacaan via Character Model (Corroborating Fast YOLO)
    char_raw, char_conf, line1_chars = read_plate_with_char_model(plate_crop, conf=0.20)
    char_raw = char_raw.upper()
    c_clean = re.sub(r'[^A-Z0-9]', '', char_raw)

    # 2b. Fallback Deskew hanya jika pembacaan PaddleOCR dan Char Model belum mendapatkan plat valid
    if not is_valid_indonesian_plate_structure(paddle_raw)[0] and (char_conf < 0.45 or len(c_clean) < 4):
        deskewed_crop, skew_angle = deskew_plate(plate_crop)
        if abs(skew_angle) >= 3.0:
            d_paddle, d_pconf, d_all_p = read_plate_with_paddleocr(deskewed_crop)
            if d_pconf > paddle_conf:
                paddle_raw, paddle_conf, all_paddle = d_paddle, d_pconf, d_all_p
            d_raw, d_conf, d_chars = read_plate_with_char_model(deskewed_crop, conf=0.20)
            if d_conf > char_conf:
                char_raw, char_conf = d_raw, d_conf

    final_formatted = refine_indonesian_plate(char_raw, paddle_raw, all_paddle)
    final_conf = max(paddle_conf, char_conf) if final_formatted else 0.0

    return {
        "final": final_formatted,
        "char_raw": char_raw,
        "char_conf": char_conf,
        "paddle_raw": paddle_raw,
        "paddle_conf": paddle_conf,
        "confidence": final_conf,
        "method": "ensemble_paddle_primary"
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


def get_front_of_camera_score(cand, plates, img_w, img_h, is_in_roi=False):
    """
    Menghitung skor prioritas kendaraan 'di depan kamera'.
    Memprioritaskan kendaraan foreground (bawah/tengah), berukuran signifikan,
    dan menaungi plat nomor yang aktif di depan kamera.
    """
    x1, y1, x2, y2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area = bw * bh
    area_ratio = area / float(max(1, img_w * img_h))

    # Objek sangat kecil (< 1.5% luas frame) yang berada jauh di luar area deteksi diabaikan
    if not is_in_roi:
        if area_ratio < 0.015:
            return -1.0
        if y2 < (img_h * 0.18):
            return -1.0

    cx = (x1 + x2) / 2.0
    norm_cx = cx / float(max(1, img_w))
    norm_y2 = y2 / float(max(1, img_h))

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
        self.last_plate_bbox = None
        self.last_plate_conf = 0.0
        
        # Controlled Tracking & ROI State
        is_inside = initial_det.get("inside_interest_area", False) if initial_det else False
        self.inside_interest_area = is_inside
        self.ever_inside_roi = is_inside
        self.frames_outside_after_inside = 0
        self.lost_interest = False
        self.lost_interest_time = None
        self.lifecycle_state = "NEW"
        self.frames_missing = 0
        self.status = "ANALYZING" if is_inside else "OUTSIDE_ROI"

        # Asynchronous Task Management State (Latest-Only Architecture)
        self.plate_pending = False
        self.pending_plate_frame = None  # (vehicle_crop, v_bbox, full_img, iw, ih, submit_time, initial_vtype, frame_id, capture_ts)
        self.last_plate_attempt_time = 0.0
        self.last_plate_update_time = 0.0
        self.plate_attempt_count = 0
        self.plate_detected = False
        self.ocr_pending = False
        self.pending_ocr_job = None      # (job_id, track_id, plate_crop, crop_q, full_img, submit_time, frame_id, capture_ts)
        self.body_pending = False
        self.last_ocr_job_id = 0
        self.last_processed_ocr_job_id = 0
        self.last_ocr_text = None
        self.last_ocr_conf = 0.0
        self.last_ocr_time = None

        # Performance Breakdown Timings (ms)
        self.vehicle_infer_ms = 0.0
        self.plate_infer_ms = 0.0
        self.ocr_infer_ms = 0.0
        self.total_processing_ms = 0.0

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

        # Candidate accumulation (persists across frames, evidence-based weighting)
        self.candidate_counts = {}
        self.candidate_scores = {}
        self.candidate_best_obs = {}
        self.best_candidate = None
        self.best_candidate_display = None
        self.best_candidate_conf = 0.0
        self.consecutive_plate = ""
        self.consecutive_count = 0
        self.valid_plate_observations = deque(maxlen=15)

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
        if is_inside and p_text:
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

    def add_ocr_result(self, job_id, text, conf, crop_q, ocr_method="ensemble"):
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
            return {
                "track_id": self.track_id,
                "ocr_candidate": self.best_candidate_display or text,
                "ocr_confidence": self.best_candidate_conf,
                "candidate_matches": self.candidate_counts.get(self.best_candidate, 0) if self.best_candidate else 0,
                "status": self.status,
                "is_confirmed": self.status in ["CONFIRMED", "HISTORY_SAVED"]
            }

        self.last_processed_ocr_job_id = max(self.last_processed_ocr_job_id, job_id)
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
                "method": ocr_method
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
        Jika hilang:
        - Finalisasi kandidat terbaik jika memenuhi syarat konfirmasi (Path A atau Path B).
        - Jika tidak memenuhi syarat: buang kandidat (DISCARDED) agar riwayat palsu tidak pernah tercatat.
        """
        if now is None:
            now = time.time()
        if self.lost_interest:
            return self.status

        time_since_seen = now - self.last_seen
        is_time_lost = time_since_seen > LOST_INTEREST_TIMEOUT_SEC
        is_left_roi = (self.ever_inside_roi and self.frames_outside_after_inside >= LOST_INTEREST_OUTSIDE_FRAMES)

        if is_time_lost or is_left_roi:
            self.lost_interest = True
            self.lost_interest_time = now
            print(f"[LOST INTEREST] Track {self.track_id} ditandai LOST (age={time_since_seen:.2f}s, outside_frames={self.frames_outside_after_inside})")

            if self.status in ["CONFIRMED", "HISTORY_SAVED"] or self.history_saved:
                return self.status

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

        return self.status

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
        # A1: Single observation with >=80% confidence and valid Samsat plate structure (min 5 chars)
        if is_struct and best_conf >= 0.80 and (len(best_cln) >= 5 or p_type == "military"):
            confirmed = True
            reason = f"PATH_A_STRONG_EVIDENCE (conf={best_conf:.2f})"

        # A2: 70-79% confidence requiring >= 2 matching observations (min 5 chars)
        elif is_struct and best_count >= 2 and best_conf >= 0.70 and (len(best_cln) >= 5 or p_type == "military"):
            confirmed = True
            reason = f"PATH_A_TWO_FRAME_MATCH (count={best_count}, conf={best_conf:.2f})"

        # ============================================================
        # PATH B — TEMPORAL CONSENSUS FOR <70% OBSERVATIONS
        # ============================================================
        # B1: Consecutive streak >= 3 with confidence >= 0.65 (min 5 chars)
        elif self.consecutive_count >= 3 and self.consecutive_plate == best_cln and best_conf >= 0.65 and (len(best_cln) >= 5 or p_type == "military"):
            confirmed = True
            reason = f"PATH_B_CONSECUTIVE_STREAK (streak={self.consecutive_count}, conf={best_conf:.2f})"

        # B2: Temporal consensus: >= 3 matching occurrences in observations with confidence >= 0.65 (min 5 chars)
        elif is_struct and best_count >= 3 and best_conf >= 0.65 and (len(best_cln) >= 5 or p_type == "military"):
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

        now_ts = time.time()
        conf_t = self.confirmation_time or now_ts
        total_e2e_ms = round((conf_t - self.created_at) * 1000.0, 1)
        v_ms = round(self.vehicle_infer_ms, 1)
        p_ms = round(self.plate_infer_ms, 1)
        o_ms = round(self.ocr_infer_ms, 1)
        total_proc_ms = total_e2e_ms if total_e2e_ms > 0 else round(v_ms + p_ms + o_ms, 1)
        self.total_processing_ms = total_proc_ms

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
            "confirmation_time": self.confirmation_time,
            "processing_ms": int(round(total_proc_ms)),
            "timing": {
                "vehicle_ms": v_ms,
                "plate_ms": p_ms,
                "ocr_ms": o_ms,
                "total_ms": total_proc_ms
            }
        }
        self.confirmation_score = round(consistency_score, 2)
        self.status = "CONFIRMED"
        self.is_locked = True
        print(f"[PERF_TOTAL] track_id={self.track_id} total_ms={total_proc_ms:.1f} (vehicle={v_ms:.1f}ms plate={p_ms:.1f}ms ocr={o_ms:.1f}ms)", flush=True)
        print(f"[ENTRY_HISTORY] track_id={self.track_id} plate=\"{stable_plate}\" processing_ms={int(round(total_proc_ms))}", flush=True)

    def get_current_stability_count(self):
        """Mengembalikan jumlah observasi plat yang cocok saat ini (untuk progress UI 1/3, 2/3, dst)."""
        if self.status in ["CONFIRMED", "HISTORY_SAVED"] or self.history_saved:
            return 3
        if self.best_candidate and self.best_candidate in self.candidate_counts:
            return min(self.candidate_counts[self.best_candidate], 3)
        if self.consecutive_count > 0:
            return min(self.consecutive_count, 3)
        return 1 if self.frames else 0

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
    Lifecycle: NEW -> ACTIVE -> EXITING -> EXPIRED
    """
    def __init__(self):
        self.tracks = {}
        self.lock = threading.RLock()
        self.next_fallback_id = 1
        self.primary_track_id = None

    def reset(self):
        with self.lock:
            self.tracks.clear()
            self.next_fallback_id = 1
            self.primary_track_id = None

    def get_or_create_track(self, raw_tid, bbox, img_w=1920, img_h=1080, frame_id=None):
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
                print(f"[TRACK_CREATED]\ntrack_id={tid}\nframe_id={frame_id}", flush=True)
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
                    trk.lifecycle_state = "EXITING"
                    print(f"[TRACK_EXITING]\ntrack_id={tid}", flush=True)
                if prev_status != "CONFIRMED" and new_status == "CONFIRMED":
                    newly_confirmed_tracks.append(trk)

                # Hapus track yang sudah lost / tidak terlihat > 1.5s agar tidak tersisa di active state
                if (trk.lost_interest or trk.lifecycle_state in ("EXITING", "LOST", "DISCARDED")) and (now - trk.last_seen) > 1.5:
                    stale_ids.append(tid)

            for tid in stale_ids:
                trk = self.tracks[tid]
                trk.lifecycle_state = "EXPIRED"
                print(f"[TRACK_EXPIRED]\ntrack_id={tid}", flush=True)
                del self.tracks[tid]

            if stale_ids:
                active_ids = [t for t, tr in self.tracks.items() if tr.lifecycle_state == "ACTIVE"]
                print(f"[ACTIVE_TRACKS]\ntrack_ids={active_ids}", flush=True)

        return newly_confirmed_tracks, newly_lost_tracks

    def _match_track(self, bbox, curr_plate=None, img_w=1920, img_h=1080, iou_thresh=0.45):
        if not bbox:
            return None
        now = time.time()

        best_id = None
        best_iou = -1.0

        for tid, trk in self.tracks.items():
            # Hanya cocokkan dengan track yang aktif dalam 0.5 detik terakhir dan TIDAK lost/exiting
            if (now - trk.last_seen) > 0.5 or trk.lost_interest or trk.lifecycle_state in ("EXITING", "LOST", "EXPIRED", "DISCARDED"):
                continue

            if trk.last_bbox:
                iou = compute_iou(bbox, trk.last_bbox)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
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
            stale_ids = [tid for tid, trk in self.tracks.items() if (now - trk.last_seen) > 1.5 and (trk.lost_interest or trk.lifecycle_state in ("EXITING", "LOST"))]
            for tid in stale_ids:
                trk = self.tracks[tid]
                trk.lifecycle_state = "EXPIRED"
                print(f"[TRACK_EXPIRED]\ntrack_id={tid}", flush=True)
                del self.tracks[tid]
            if stale_ids:
                active_ids = [t for t, tr in self.tracks.items() if tr.lifecycle_state == "ACTIVE"]
                print(f"[ACTIVE_TRACKS]\ntrack_ids={active_ids}", flush=True)

            if not detections:
                if self.primary_track_id is not None:
                    print(f"[PRIMARY_SWITCH]\nold_track_id={self.primary_track_id}\nnew_track_id=None", flush=True)
                    self.primary_track_id = None
                for tid, trk in self.tracks.items():
                    if trk.lifecycle_state == "ACTIVE":
                        trk.lifecycle_state = "EXITING"
                        print(f"[TRACK_EXITING]\ntrack_id={tid}", flush=True)
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
            visible_tids = set()
            for det in detections:
                raw_tid = det.get("track_id")
                bbox = det.get("bbox")
                iw = det.get("image_width", 1920)
                ih = det.get("image_height", 1080)

                if raw_tid is not None:
                    tid = raw_tid
                else:
                    matched_id = self._match_track(bbox, img_w=iw, img_h=ih)
                    if matched_id is not None:
                        tid = matched_id
                    else:
                        tid = self.next_fallback_id
                        self.next_fallback_id += 1

                det["track_id"] = tid
                visible_tids.add(tid)

                if tid not in self.tracks:
                    self.tracks[tid] = VehicleTrack(tid, det)
                    track = self.tracks[tid]
                    print(f"[TRACK_CREATED]\ntrack_id={tid}", flush=True)
                else:
                    track = self.tracks[tid]
                    track.add_frame(det)

                track.frames_missing = 0
                if track.lifecycle_state != "ACTIVE" and not track.lost_interest:
                    track.lifecycle_state = "ACTIVE"
                    print(f"[TRACK_ACTIVE]\ntrack_id={tid}", flush=True)

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

            # Tandai track yang tidak terdeteksi di frame ini sebagai EXITING
            for tid, trk in self.tracks.items():
                if tid not in visible_tids:
                    trk.frames_missing = getattr(trk, 'frames_missing', 0) + 1
                    if trk.lifecycle_state == "ACTIVE":
                        trk.lifecycle_state = "EXITING"
                        print(f"[TRACK_EXITING]\ntrack_id={tid}", flush=True)

            # Primary Track Switching Tracking
            active_candidates = [d for d in updated_detections if d.get("inside_interest_area") and not d.get("lost_interest") and d.get("status") != "OUTSIDE_ROI"]
            new_primary_id = active_candidates[0].get("track_id") if active_candidates else None
            if new_primary_id != self.primary_track_id:
                print(f"[PRIMARY_SWITCH]\nold_track_id={self.primary_track_id}\nnew_track_id={new_primary_id}", flush=True)
                self.primary_track_id = new_primary_id

            return updated_detections


def _async_body_task(track_id, full_img, bbox, initial_vtype, v_conf, has_bus_det, body_hints, frame_id=None, capture_timestamp=None):
    """Worker task untuk asynchronous vehicle body classification di background thread pool."""
    try:
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if not track or getattr(track, 'lifecycle_state', '') in ('EXITING', 'LOST', 'EXPIRED', 'DISCARDED') or track.lost_interest:
                print(f"[STALE_RESULT_DISCARDED]\ntrack_id={track_id}\nframe_id={frame_id}\nreason=TRACK_INACTIVE", flush=True)
                return

        vehicle_type, body_style, body_style_conf = classify_vehicle_indonesian(
            full_img, bbox, initial_vtype, v_conf, has_bus_det=has_bus_det, body_hints=body_hints
        )
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if not track or getattr(track, 'lifecycle_state', '') in ('EXITING', 'LOST', 'EXPIRED', 'DISCARDED') or track.lost_interest:
                print(f"[STALE_RESULT_DISCARDED]\ntrack_id={track_id}\nframe_id={frame_id}\nreason=TRACK_INACTIVE", flush=True)
                return
            track.cached_body_style = body_style
            track.cached_body_conf = body_style_conf
            track.cached_vtype = vehicle_type
            track.body_pending = False
            if track.confirmed_data:
                track.confirmed_data["body_style"] = body_style
                track.confirmed_data["body_style_confidence"] = round(body_style_conf, 3) if body_style_conf else None
                track.confirmed_data["vehicle_type"] = vehicle_type

        # Siarkan update body style via WebSocket
        ws_broadcaster.broadcast({
            "type": "body_update",
            "frame_id": frame_id,
            "capture_timestamp": capture_timestamp,
            "track_id": track_id,
            "vehicle_type": vehicle_type,
            "body_style": body_style,
            "body_style_confidence": round(body_style_conf, 3) if body_style_conf else None,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"[ASYNC BODY ERROR] Track {track_id}: {e}")
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if track:
                track.body_pending = False


def _async_plate_task(track_id, vehicle_crop, v_bbox, full_img, iw, ih, submit_time, initial_vtype="car", frame_id=None, capture_timestamp=None):
    """Worker task untuk asynchronous Plate Detection di background thread pool."""
    t_plate_start = time.time()
    print(f"[PLATE_INFER_START]\ntrack_id={track_id}\nframe_id={frame_id}", flush=True)
    try:
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if not track or getattr(track, 'lifecycle_state', '') in ('EXITING', 'LOST', 'EXPIRED', 'DISCARDED') or track.lost_interest:
                print(f"[STALE_RESULT_DISCARDED]\ntrack_id={track_id}\nframe_id={frame_id}\nreason=TRACK_INACTIVE", flush=True)
                return

        vx1, vy1, vx2, vy2 = v_bbox
        vh = max(1, vy2 - vy1)
        plate_conf_val = None
        abs_plate_bbox = None
        plate_crop = np.array([])

        if vehicle_crop.size > 0:
            pdet_crop = plate_detector.predict(vehicle_crop, conf=0.08, imgsz=PLATE_INFER_IMGSZ, device=DEVICE, verbose=False)[0]
            valid_crops = []
            for b in pdet_crop.boxes:
                cls_id = int(b.cls[0])
                cname = plate_detector.names.get(cls_id, "")
                if cname not in ['plat-nomor', 'license_plate'] and len(plate_detector.names) > 1:
                    continue
                cpx1, cpy1, cpx2, cpy2 = map(int, b.xyxy[0].tolist())
                abs_box = [vx1 + cpx1, vy1 + cpy1, vx1 + cpx2, vy1 + cpy2]
                if is_valid_plate_box(abs_box, iw, ih):
                    p_conf = float(b.conf[0])
                    rel_y = (cpy1 + cpy2) / (2.0 * vh)
                    # Filter out hood/windshield false positives for cars/trucks (rel_y < 0.32)
                    if initial_vtype != "motorcycle" and rel_y < 0.32:
                        continue
                    # Prioritas posisi bumper bawah kendaraan (rel_y >= 0.60): plat bumper mendapat bobot 2.4x
                    y_factor = 2.4 if rel_y >= 0.60 else (1.1 if rel_y >= 0.48 else 0.5)
                    score = p_conf * y_factor
                    valid_crops.append((abs_box, p_conf, score))

            if not valid_crops and vehicle_crop.size > 0:
                pdet_legacy = plate_model_legacy.predict(vehicle_crop, conf=0.06, device=DEVICE, verbose=False)[0]
                for b in pdet_legacy.boxes:
                    cpx1, cpy1, cpx2, cpy2 = map(int, b.xyxy[0].tolist())
                    abs_box = [vx1 + cpx1, vy1 + cpy1, vx1 + cpx2, vy1 + cpy2]
                    if is_valid_plate_box(abs_box, iw, ih):
                        p_conf = float(b.conf[0])
                        rel_y = (cpy1 + cpy2) / (2.0 * vh)
                        if initial_vtype != "motorcycle" and rel_y < 0.32:
                            continue
                        y_factor = 2.4 if rel_y >= 0.60 else (1.1 if rel_y >= 0.48 else 0.5)
                        score = p_conf * y_factor
                        valid_crops.append((abs_box, p_conf, score))

            if valid_crops:
                valid_crops.sort(key=lambda x: -x[2])
                abs_plate_bbox, plate_conf_val, _ = valid_crops[0]
                plate_crop = crop_plate_with_padding(full_img, abs_plate_bbox[0], abs_plate_bbox[1],
                                                     abs_plate_bbox[2], abs_plate_bbox[3])

        t_plate_end = time.time()
        plate_duration_ms = (t_plate_end - t_plate_start) * 1000.0
        print(f"[PLATE_INFER_END]\ntrack_id={track_id}\nframe_id={frame_id}\nduration_ms={plate_duration_ms:.1f}\nfound={abs_plate_bbox is not None}", flush=True)

        ocr_to_submit = None
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if not track or getattr(track, 'lifecycle_state', '') in ('EXITING', 'LOST', 'EXPIRED', 'DISCARDED') or track.lost_interest:
                print(f"[STALE_RESULT_DISCARDED]\ntrack_id={track_id}\nframe_id={frame_id}\nreason=TRACK_INACTIVE", flush=True)
                return
            track.plate_infer_ms = plate_duration_ms
            if abs_plate_bbox is not None:
                track.last_plate_bbox = abs_plate_bbox
                track.last_plate_conf = plate_conf_val
                track.last_plate_update_time = time.time()
                track.plate_detected = True

                # Queue or prepare OCR job if valid crop exists and track is not locked
                if plate_crop.size > 0 and not track.is_locked:
                    crop_q = compute_crop_quality(plate_crop, plate_conf_val)
                    if not track.ocr_pending:
                        job_id = get_next_ocr_job_id()
                        track.ocr_pending = True
                        track.last_ocr_job_id = job_id
                        ocr_to_submit = (job_id, track_id, plate_crop, crop_q, full_img, time.time(), frame_id, capture_timestamp)
                    else:
                        # Replace pending OCR crop with latest crop
                        job_id = get_next_ocr_job_id()
                        track.pending_ocr_job = (job_id, track_id, plate_crop, crop_q, full_img, time.time(), frame_id, capture_timestamp)

        if abs_plate_bbox is not None:
            t_plate_sent = time.time()
            # FAST PLATE BBOX: Broadcast immediately to WebSocket without waiting for OCR!
            ws_broadcaster.broadcast({
                "type": "plate_update",
                "frame_id": frame_id,
                "capture_timestamp": capture_timestamp,
                "track_id": track_id,
                "plate_bbox": abs_plate_bbox,
                "plate_confidence": round(plate_conf_val, 3),
                "plate_detected": True,
                "status": track.status,
                "ocr_status": "SEARCHING" if not track.best_candidate else track.status,
                "inside_interest_area": True,
                "lost_interest": False,
                "timestamp": t_plate_sent
            })

            if ocr_to_submit is not None:
                ocr_executor.submit(_async_ocr_task, *ocr_to_submit)
    except Exception as e:
        print(f"[ASYNC PLATE ERROR] Track {track_id}: {e}")
    finally:
        # Check if there is a newer pending plate frame to process (Latest-Only Plate Processing)
        next_plate_job = None
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if track:
                if not track.lost_interest and not track.is_locked and getattr(track, 'lifecycle_state', '') == 'ACTIVE' and track.pending_plate_frame is not None:
                    next_plate_job = track.pending_plate_frame
                    track.pending_plate_frame = None
                    track.plate_pending = True
                else:
                    track.plate_pending = False

        if next_plate_job is not None:
            vcrop, vbx, fimg, i_w, i_h, n_time, ivtype, f_id, c_ts = next_plate_job
            plate_executor.submit(_async_plate_task, track_id, vcrop, vbx, fimg, i_w, i_h, n_time, ivtype, f_id, c_ts)


def _async_ocr_task(job_id, track_id, crop, crop_q, full_img, submit_time, frame_id=None, capture_timestamp=None):
    """Worker task untuk asynchronous PaddleOCR & ensemble plate reading di thread pool."""
    t_ocr_start = time.time()
    print(f"[OCR_START]\ntrack_id={track_id}\nframe_id={frame_id}", flush=True)
    try:
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if not track or getattr(track, 'lifecycle_state', '') in ('EXITING', 'LOST', 'EXPIRED', 'DISCARDED') or track.lost_interest:
                print(f"[STALE_RESULT_DISCARDED]\ntrack_id={track_id}\nframe_id={frame_id}\nreason=TRACK_INACTIVE", flush=True)
                return

        res = ensemble_plate_reading(crop)
        t_ocr_end = time.time()
        ocr_duration_ms = (t_ocr_end - t_ocr_start) * 1000.0
        text = res.get("final")
        conf = float(res.get("confidence", 0.0) or 0.0)
        method = res.get("method", "ensemble")
        print(f"[OCR_END]\ntrack_id={track_id}\nframe_id={frame_id}\ntext={text}\nconfidence={conf:.2f}\nduration_ms={ocr_duration_ms:.1f}", flush=True)

        is_newly_confirmed = False
        confirmed_data = None
        update_info = None

        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if not track or getattr(track, 'lifecycle_state', '') in ('EXITING', 'LOST', 'EXPIRED', 'DISCARDED') or track.lost_interest:
                print(f"[STALE_RESULT_DISCARDED]\ntrack_id={track_id}\nframe_id={frame_id}\nreason=TRACK_INACTIVE", flush=True)
                return
            track.ocr_infer_ms = ocr_duration_ms
            prev_status = track.status
            update_info = track.add_ocr_result(job_id, text, conf, crop_q, ocr_method=method)

            if prev_status != "CONFIRMED" and track.status == "CONFIRMED":
                is_newly_confirmed = True
                confirmed_data = dict(track.confirmed_data) if track.confirmed_data else None
                track.history_saved = True
                track.status = "HISTORY_SAVED"
                track.is_locked = True
                t_conf = time.time()
                print(f"[PERF_LATENCY] [CONFIRMATION: track_id={track_id} plate='{confirmed_data.get('license_plate')}' time={t_conf:.3f} reason='{track.confirmation_reason}']", flush=True)

        # Siarkan pembaruan OCR langsung ke WebSocket tanpa menunggu konfirmasi
        if update_info:
            cand_count = update_info.get("candidate_matches", 0)
            ocr_stat = "CONFIRMED" if update_info["is_confirmed"] else ("GOOD_CANDIDATE" if cand_count >= 2 else "ANALYZING")
            ws_broadcaster.broadcast({
                "type": "ocr_update",
                "frame_id": frame_id,
                "capture_timestamp": capture_timestamp,
                "track_id": track_id,
                "license_plate": update_info["ocr_candidate"],
                "ocr_candidate": update_info["ocr_candidate"],
                "ocr_confidence": round(update_info["ocr_confidence"], 3) if update_info.get("ocr_confidence") else None,
                "candidate_matches": cand_count,
                "ocr_status": ocr_stat,
                "status": update_info["status"],
                "lost_interest": False,
                "inside_interest_area": True,
                "timestamp": time.time()
            })

        if is_newly_confirmed and confirmed_data:
            rec = save_parking_record(confirmed_data, source_img=full_img)
    except Exception as e:
        print(f"[ASYNC OCR ERROR] Track {track_id} job {job_id}: {e}")
    finally:
        # Check if there is a newer pending OCR job to process (Latest-Only OCR Processing)
        next_ocr_job = None
        with confirmation_manager.lock:
            track = confirmation_manager.tracks.get(track_id)
            if track:
                if not track.lost_interest and not track.is_locked and getattr(track, 'lifecycle_state', '') == 'ACTIVE' and track.status not in ("CONFIRMED", "HISTORY_SAVED") and track.pending_ocr_job is not None:
                    next_ocr_job = track.pending_ocr_job
                    track.pending_ocr_job = None
                    track.ocr_pending = True
                else:
                    track.ocr_pending = False

        if next_ocr_job is not None:
            ocr_executor.submit(_async_ocr_task, *next_ocr_job)



confirmation_manager = VehicleConfirmationManager()


def run_anpr(image_input, vehicle_conf=None, motorcycle_conf=None, plate_conf=None, single_vehicle_mode=True, is_stream=False, frame_id=None, capture_timestamp=None):
    t_start = time.time()
    if frame_id is not None:
        print(f"[FRAME_RECEIVED]\nframe_id={frame_id}", flush=True)
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
    print(f"[VEHICLE_INFER_START]\nframe_id={frame_id}", flush=True)
    infer_conf = min(v_conf_thresh, m_conf_thresh)
    if is_stream:
        fut_v = ai_pool.submit(vehicle_model.track, img, persist=True, tracker="bytetrack.yaml",
                               conf=infer_conf, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    else:
        fut_v = ai_pool.submit(vehicle_model.predict, img, conf=infer_conf,
                               imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    vdet = fut_v.result()[0]
    t_yolo_end = time.time()
    t_yolo = t_yolo_end - t_yolo_0
    print(f"[VEHICLE_INFER_END]\nframe_id={frame_id}\nduration_ms={t_yolo*1000:.1f}", flush=True)

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
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area = bw * bh
        # Minimum physical box size check: ignore noise smaller than 40x40 or < 1.5% of frame area
        if bw < 40 or bh < 40 or (area / float(max(1, iw * ih))) < 0.015:
            continue

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

    # 2b. FILTER & FOKUS KENDARAAN DI AREA DETEKTOR (Detection Area Priority)
    scored_candidates = []
    if candidates:
        for c in candidates:
            inside, overlap = is_vehicle_inside_roi(c["x1"], c["y1"], c["x2"], c["y2"], iw, ih)
            c["_inside_roi"] = inside
            c["_roi_overlap"] = overlap
            s = get_front_of_camera_score(c, global_plates, iw, ih, is_in_roi=inside)
            if s > 0 or inside:
                c["front_score"] = s if s > 0 else 1.0
                scored_candidates.append(c)

        # PRIORITAS TINGGI: Kendaraan yang berada DI DALAM AREA DETEKTOR selalu di urutan terdepan (index 0)!
        # Jika ada beberapa kendaraan di ROI, prioritaskan yang paling depan/dekat ke palang gate
        scored_candidates.sort(key=lambda c: (-int(c.get("_inside_roi", False)), -c.get("front_score", 0.0)))

        if single_vehicle_mode and not is_stream:
            candidates = [scored_candidates[0]] if scored_candidates else []
        else:
            candidates = scored_candidates

    print(f"[VEHICLE_RESULT] frame_id={frame_id} count={len(candidates)} active_track_id={candidates[0]['track_id'] if candidates else None}", flush=True)
    any_inside_roi = any(c.get("_inside_roi") for c in candidates)

    # Single Photo Mode: Jalankan deteksi plat global synchronous untuk akurasi foto
    if not is_stream and (any_inside_roi or not candidates):
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
                cand_track_id, [x1, y1, x2, y2], img_w=iw, img_h=ih, frame_id=frame_id
            )
            cand_track_id = tid
            t_yolo_ms = (t_yolo_end - t_yolo_0) * 1000.0
            existing_trk.vehicle_infer_ms = t_yolo_ms
            print(f"[PERF_VEHICLE] frame_id={frame_id} track_id={tid} duration_ms={t_yolo_ms:.1f}", flush=True)

            # Cek apakah ada deteksi 'bus' di area kendaraan ini
            has_bus_det = any(
                map_vehicle_class_name(b.cls[0], vehicle_model) == 'bus' and
                compute_iou([x1, y1, x2, y2], list(map(int, b.xyxy[0].tolist()))) > 0.30
                for b in vdet.boxes
            )

            # JIKA DI LUAR DETECTION AREA ATAU LOST INTEREST:
            if (not is_in_roi) or existing_trk.lost_interest:
                results_out.append({
                    "track_id": cand_track_id,
                    "vehicle_type": existing_trk.cached_vtype or initial_vtype,
                    "body_style": existing_trk.cached_body_style,
                    "body_style_confidence": round(existing_trk.cached_body_conf, 3) if existing_trk.cached_body_conf else None,
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
            if is_stream:
                # ============================================================
                # FAST NON-BLOCKING STREAM PATH:
                # Vehicle bounding box renders immediately (< 35ms)
                # Body classifier and Plate/PaddleOCR run in background threads!
                # ============================================================
                if not existing_trk.cached_body_style and not existing_trk.body_pending and not existing_trk.lost_interest:
                    existing_trk.body_pending = True
                    body_executor.submit(_async_body_task, cand_track_id, img.copy(), [x1, y1, x2, y2],
                                         initial_vtype, v_conf, has_bus_det, body_hints, frame_id, capture_timestamp)

                if not existing_trk.is_locked and not existing_trk.lost_interest:
                    now_plate = time.time()
                    # 1. Asynchronous Plate Detection: ONE active job + ONE latest pending frame
                    with confirmation_manager.lock:
                        if not existing_trk.plate_pending:
                            existing_trk.plate_pending = True
                            existing_trk.last_plate_attempt_time = now_plate
                            existing_trk.plate_attempt_count += 1
                            plate_executor.submit(
                                _async_plate_task,
                                cand_track_id,
                                vehicle_crop.copy(),
                                [x1, y1, x2, y2],
                                img.copy(),
                                iw,
                                ih,
                                now_plate,
                                initial_vtype,
                                frame_id,
                                capture_timestamp
                            )
                        else:
                            # Replace pending frame with the latest one (discarding older frames)
                            existing_trk.pending_plate_frame = (
                                vehicle_crop.copy(),
                                [x1, y1, x2, y2],
                                img.copy(),
                                iw,
                                ih,
                                now_plate,
                                initial_vtype,
                                frame_id,
                                capture_timestamp
                            )

                    # 2. Asynchronous OCR refinement jika plat sudah terdeteksi
                    if existing_trk.last_plate_bbox and (time.time() - (existing_trk.last_ocr_time or 0) > 0.15 or existing_trk.best_candidate is None):
                        p_crop = crop_plate_with_padding(img, existing_trk.last_plate_bbox[0], existing_trk.last_plate_bbox[1],
                                                         existing_trk.last_plate_bbox[2], existing_trk.last_plate_bbox[3])
                        crop_q = compute_crop_quality(p_crop, existing_trk.last_plate_conf)
                        ocr_to_submit = None
                        with confirmation_manager.lock:
                            if not existing_trk.ocr_pending:
                                job_id = get_next_ocr_job_id()
                                existing_trk.ocr_pending = True
                                existing_trk.last_ocr_job_id = job_id
                                ocr_to_submit = (job_id, cand_track_id, p_crop, crop_q, img.copy(), time.time(), frame_id, capture_timestamp)
                            else:
                                job_id = get_next_ocr_job_id()
                                existing_trk.pending_ocr_job = (job_id, cand_track_id, p_crop, crop_q, img.copy(), time.time(), frame_id, capture_timestamp)

                        if ocr_to_submit is not None:
                            ocr_executor.submit(_async_ocr_task, *ocr_to_submit)

                vehicle_type = existing_trk.cached_vtype or initial_vtype
                body_style = existing_trk.cached_body_style
                body_style_conf = existing_trk.cached_body_conf
                abs_plate_bbox = existing_trk.last_plate_bbox
                plate_conf_val = existing_trk.last_plate_conf

                if existing_trk.is_locked or existing_trk.history_saved:
                    plate_text = existing_trk.confirmed_data.get("license_plate") if existing_trk.confirmed_data else existing_trk.best_candidate_display
                    ocr_conf = existing_trk.confirmed_data.get("plate_confidence", 0.0) if existing_trk.confirmed_data else existing_trk.best_candidate_conf
                    ocr_method = "locked_confirmed"
                else:
                    plate_text = existing_trk.best_candidate_display
                    ocr_conf = existing_trk.best_candidate_conf or 0.0
                    ocr_method = "temporal_best_candidate" if plate_text else None

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
                    "plate_crop": None,
                    "image_width": iw,
                    "image_height": ih,
                    "inside_interest_area": True,
                    "plate_detected": bool(abs_plate_bbox is not None)
                })

            else:
                # Single Photo Mode: Synchronous body style + plate OCR
                t_b0 = time.time()
                vehicle_type, body_style, body_style_conf = classify_vehicle_indonesian(
                    img, [x1, y1, x2, y2], initial_vtype, v_conf, has_bus_det=has_bus_det, body_hints=body_hints
                )
                t_body_total += (time.time() - t_b0)

                plate_conf_val = None
                abs_plate_bbox = None
                plate_crop = np.array([])

                if vehicle_crop.size > 0:
                    vh = max(1, y2 - y1)
                    pdet_crop = plate_detector.predict(vehicle_crop, conf=0.08, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
                    valid_crops = []
                    for b in pdet_crop.boxes:
                        cls_id = int(b.cls[0])
                        cname = plate_detector.names.get(cls_id, "")
                        if cname not in ['plat-nomor', 'license_plate'] and len(plate_detector.names) > 1:
                            continue
                        cpx1, cpy1, cpx2, cpy2 = map(int, b.xyxy[0].tolist())
                        abs_box = [x1 + cpx1, y1 + cpy1, x1 + cpx2, y1 + cpy2]
                        if is_valid_plate_box(abs_box, iw, ih):
                            p_conf = float(b.conf[0])
                            rel_y = (cpy1 + cpy2) / (2.0 * vh)
                            if initial_vtype != "motorcycle" and rel_y < 0.32:
                                continue
                            y_factor = 2.4 if rel_y >= 0.60 else (1.1 if rel_y >= 0.48 else 0.5)
                            score = p_conf * y_factor
                            valid_crops.append((abs_box, p_conf, score))
                    if not valid_crops:
                        pdet_legacy = plate_model_legacy.predict(vehicle_crop, conf=0.06, device=DEVICE, verbose=False)[0]
                        for b in pdet_legacy.boxes:
                            cpx1, cpy1, cpx2, cpy2 = map(int, b.xyxy[0].tolist())
                            abs_box = [x1 + cpx1, y1 + cpy1, x1 + cpx2, y1 + cpy2]
                            if is_valid_plate_box(abs_box, iw, ih):
                                p_conf = float(b.conf[0])
                                rel_y = (cpy1 + cpy2) / (2.0 * vh)
                                if initial_vtype != "motorcycle" and rel_y < 0.32:
                                    continue
                                y_factor = 2.4 if rel_y >= 0.60 else (1.1 if rel_y >= 0.48 else 0.5)
                                score = p_conf * y_factor
                                valid_crops.append((abs_box, p_conf, score))
                    if valid_crops:
                        valid_crops.sort(key=lambda x: -x[2])
                        abs_plate_bbox, plate_conf_val, _ = valid_crops[0]
                        plate_crop = crop_plate_with_padding(img, abs_plate_bbox[0], abs_plate_bbox[1],
                                                             abs_plate_bbox[2], abs_plate_bbox[3])

                plate_text = None
                ocr_method = None
                ocr_conf = 0.0

                if plate_crop.size > 0:
                    t_o0 = time.time()
                    ensemble_res = ensemble_plate_reading(plate_crop)
                    plate_text = ensemble_res.get("final")
                    ocr_method = ensemble_res.get("method")
                    ocr_conf = float(ensemble_res.get("confidence", 0.0) or 0.0)
                    t_ocr_total += (time.time() - t_o0)

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
                    "plate_crop": None,
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
CORS(app, resources={r"/*": {"origins": "*"}})
sock = Sock(app)


# ============================================================
# REAL-TIME BACKGROUND STREAM INFERENCE WORKER
# Memproses frame camera_stream_manager dan menyiarkan hasil deteksi via WebSocket secara live.
# ============================================================
class StreamInferenceWorker:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
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
        old_thread = self.thread
        self.thread = None
        if old_thread and old_thread.is_alive() and old_thread != threading.current_thread():
            old_thread.join(timeout=0.5)
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
                result = run_anpr(frame, is_stream=True, frame_id=frame_id, capture_timestamp=frame_capture_time)
                det_time_ms = round((time.time() - t0) * 1000)

                # Evaluasi lost tracks (finalize or discard; save_parking_record already broadcasts)
                lost_confirmed, newly_lost = confirmation_manager.check_lost_tracks()
                for trk in lost_confirmed:
                    if trk.confirmed_data and not trk.history_saved:
                        rec = save_parking_record(trk.confirmed_data, source_img=frame)
                        trk.history_saved = True
                        trk.status = "HISTORY_SAVED"

                detections = result.get("detections", [])
                det_ids = [d.get("track_id") for d in detections if d.get("track_id") is not None]

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

                # Check newly confirmed (save_parking_record broadcasts entry_confirmed once)
                for det in detections:
                    if det.get("is_newly_confirmed"):
                        rec = save_parking_record(det, source_img=frame)
                        det["record"] = rec

                # Broadcast detection update ke seluruh WebSocket client
                now_broadcast = time.time()
                frame_age_ms = int(round((now_broadcast - frame_capture_time) * 1000)) if frame_capture_time else det_time_ms
                payload = {
                    "type": "detection_update",
                    "frame_id": frame_id,
                    "timestamp": t0,
                    "capture_timestamp": frame_capture_time,
                    "frame_age_ms": frame_age_ms,
                    "latency_ms": det_time_ms,
                    "detections": detections,
                    "interest_area": INTEREST_AREA,
                    "image_width": result.get("image_width", 1920),
                    "image_height": result.get("image_height", 1080),
                }
                self.last_payload = payload
                ws_broadcaster.broadcast(payload)
                print(f"[DETECTION_SENT]\nframe_id={frame_id}\nactive_tracks={det_ids}", flush=True)
                print(f"[PERF_TOTAL]\nframe_id={frame_id}\nduration_ms={det_time_ms}", flush=True)
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
                        client_frame_id = msg.get("frame_id")
                        client_capture_ts = msg.get("capture_timestamp")
                        b64_img = msg.get("image", "")
                        if "," in b64_img:
                            b64_img = b64_img.split(",", 1)[1]
                        if b64_img:
                            img_bytes = base64.b64decode(b64_img)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                t0 = time.time()
                                res = run_anpr(frame, is_stream=True, frame_id=client_frame_id, capture_timestamp=client_capture_ts)
                                det_ms = round((time.time() - t0) * 1000)

                                lost_confirmed, newly_lost = confirmation_manager.check_lost_tracks()
                                for trk in lost_confirmed:
                                    if trk.confirmed_data and not trk.history_saved:
                                        rec = save_parking_record(trk.confirmed_data, source_img=frame)
                                        trk.history_saved = True
                                        trk.status = "HISTORY_SAVED"

                                detections = res.get("detections", [])
                                det_ids = [d.get("track_id") for d in detections if d.get("track_id") is not None]
                                now_ts = time.time()
                                with confirmation_manager.lock:
                                    for det in detections:
                                        trk = confirmation_manager.tracks.get(det.get("track_id"))
                                        if trk:
                                             det["lost_interest"] = trk.lost_interest
                                             det["last_seen"] = round(trk.last_seen, 3)
                                             det["last_seen_age_ms"] = int(round(max(0.0, now_ts - trk.last_seen) * 1000))

                                for d in detections:
                                    if d.get("is_newly_confirmed"):
                                        rec = save_parking_record(d, source_img=frame)
                                        d["record"] = rec

                                now_broadcast = time.time()
                                frame_age_ms = int(round((now_broadcast - client_capture_ts) * 1000)) if client_capture_ts else det_ms
                                payload = {
                                    "type": "detection_update",
                                    "frame_id": client_frame_id,
                                    "capture_timestamp": client_capture_ts,
                                    "frame_age_ms": frame_age_ms,
                                    "timestamp": t0,
                                    "latency_ms": det_ms,
                                    "detections": detections,
                                    "interest_area": INTEREST_AREA,
                                    "image_width": res.get("image_width", frame.shape[1]),
                                    "image_height": res.get("image_height", frame.shape[0]),
                                }
                                ws.send(json.dumps(payload))
                                print(f"[DETECTION_SENT]\nframe_id={client_frame_id}\nactive_tracks={det_ids}", flush=True)
                                print(f"[PERF_TOTAL]\nframe_id={client_frame_id}\nduration_ms={det_ms}", flush=True)
                except Exception as ex:
                    print(f"[WS MSG ERROR]: {ex}")
    except Exception:
        pass
    finally:
        ws_broadcaster.unregister(ws)


@app.route("/api/config/interest_area", methods=["GET", "POST", "OPTIONS"])
def config_interest_area():
    """Mengambil atau memperbarui konfigurasi polygon Detection / Interest Area."""
    global INTEREST_AREA
    if request.method == "OPTIONS":
        return "", 200

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        incoming_points = data.get("points")

        try:
            if isinstance(incoming_points, list) and len(incoming_points) >= 3:
                points = []
                for p in incoming_points:
                    if isinstance(p, dict):
                        px, py = p.get("x"), p.get("y")
                    else:
                        px, py = p[0], p[1]
                    points.append([
                        round(max(0.0, min(1.0, float(px))), 4),
                        round(max(0.0, min(1.0, float(py))), 4),
                    ])

                bounds = _roi_bounds_from_points(points)
                INTEREST_AREA = {
                    "x_min": round(bounds["x_min"], 4),
                    "y_min": round(bounds["y_min"], 4),
                    "x_max": round(bounds["x_max"], 4),
                    "y_max": round(bounds["y_max"], 4),
                    "points": points,
                }
            else:
                # Backward compatibility dengan frontend lama yang mengirim rectangle.
                x_min = max(0.0, min(1.0, float(data.get("x_min", INTEREST_AREA["x_min"]))))
                y_min = max(0.0, min(1.0, float(data.get("y_min", INTEREST_AREA["y_min"]))))
                x_max = max(0.0, min(1.0, float(data.get("x_max", INTEREST_AREA["x_max"]))))
                y_max = max(0.0, min(1.0, float(data.get("y_max", INTEREST_AREA["y_max"]))))
                if x_max <= x_min or y_max <= y_min:
                    raise ValueError("x_max/y_max must be greater than x_min/y_min")
                INTEREST_AREA = {
                    "x_min": round(x_min, 4),
                    "y_min": round(y_min, 4),
                    "x_max": round(x_max, 4),
                    "y_max": round(y_max, 4),
                    "points": [
                        [x_min, y_min], [x_max, y_min],
                        [x_max, y_max], [x_min, y_max]
                    ],
                }

            ws_broadcaster.broadcast({
                "type": "interest_area_update",
                "interest_area": INTEREST_AREA
            })
            return jsonify({"status": "ok", "interest_area": INTEREST_AREA})
        except Exception as ex:
            return jsonify({"error": f"Invalid ROI polygon: {ex}"}), 400

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
    return send_from_directory(config.FRONTEND_DIR, config.DASHBOARD_FILE)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Server ANPR aktif",
        "memory_records_count": len(LATEST_RECORDS),
        "stream_status": camera_stream_manager.get_status(),
        "interest_area": INTEREST_AREA
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
    
    # Hentikan stream kamera/video yang sedang berjalan agar lock file dilepas
    camera_stream_manager.stop()
    time.sleep(0.1)

    dest_path = config.UPLOADED_VIDEO_PATH
    try:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                dest_path = os.path.join(config.BASE_DIR, f"uploaded_{int(time.time())}.mp4")
        file.save(dest_path)
    except Exception:
        dest_path = os.path.join(config.BASE_DIR, f"uploaded_{int(time.time())}.mp4")
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

            temp_path = config.TEMP_UPLOAD_PATH
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
                        temp_path = config.TEMP_UPLOAD_PATH
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

        temp_path = config.TEMP_UPLOAD_PATH
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
        temp_path = config.TEMP_UPLOAD_PATH
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
