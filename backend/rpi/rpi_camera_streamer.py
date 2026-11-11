"""
Script Streaming & Snapshot Kamera untuk Raspberry Pi
Jalankan file ini di Raspberry Pi Anda:
    python3 backend/rpi/rpi_camera_streamer.py

Mendukung:
1. Video Stream MJPEG: http://<IP-Raspberry-Pi>:8080/video_feed
2. Snapshot Jepret Foto: http://<IP-Raspberry-Pi>:8080/capture_image
"""

import os
import hmac
import cv2
import threading
from flask import Flask, Response, jsonify, request

app = Flask(__name__)


def _check_streamer_token():
    expected = os.environ.get("ANPR_API_TOKEN", "").strip()
    if not expected:
        return True
    auth_hdr = request.headers.get("Authorization", "").strip()
    if auth_hdr.lower().startswith("bearer "):
        tok = auth_hdr[7:].strip()
    else:
        tok = request.headers.get("X-API-Token", "").strip()
    return bool(tok and hmac.compare_digest(tok, expected))


@app.before_request
def enforce_streamer_auth():
    if request.path in ("/", "/ping"):
        return None
    if not _check_streamer_token():
        return jsonify({"error": "Unauthorized: ANPR_API_TOKEN required"}), 401
    return None

# Otomatis cari index kamera yang aktif (0, 1, atau 2)
camera = None
for cam_idx in [0, 1, 2]:
    cap = cv2.VideoCapture(cam_idx)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            camera = cap
            print(f"✅ Kamera Raspberry Pi aktif di device index {cam_idx}")
            break
        cap.release()

if camera is None:
    print("⚠️ Membuka default kamera index 0...")
    camera = cv2.VideoCapture(0)

# Set resolusi streaming optimal untuk WiFi
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

camera_lock = threading.Lock()

def read_frame():
    """Mengambil 1 frame secara thread-safe."""
    with camera_lock:
        if not camera.isOpened():
            return False, None
        return camera.read()

def generate_mjpeg():
    """Mengambil frame dari kamera dan membroadcast dalam format MJPEG stream."""
    while True:
        success, frame = read_frame()
        if not success or frame is None:
            continue
        
        # Kompres ke JPEG kualitas 80% (ringan dan jernih)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return "<h3>Kamera Raspberry Pi ANPR Aktif!</h3><p>Stream: <a href='/video_feed'>/video_feed</a> | Snapshot: <a href='/capture_image'>/capture_image</a></p>"

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "online", "code": 200, "device": "Raspberry Pi Camera Streamer"})

@app.route('/video_feed')
def video_feed():
    """Endpoint HTTP MJPEG Stream untuk web dashboard ANPR."""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['GET', 'POST'])
@app.route('/capture', methods=['GET', 'POST'])
def capture_snapshot():
    """Endpoint Snapshot 1 foto untuk tombol Capture & Scan di dashboard."""
    success, frame = read_frame()
    if not success or frame is None:
        return Response("Gagal membaca kamera", status=500, mimetype="text/plain")
    
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ret:
        return Response("Gagal encode JPEG", status=500, mimetype="text/plain")
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    print("=" * 60)
    print("🎥 Raspberry Pi Camera Streamer & Snapshot Aktif!")
    print("Stream URL:   http://<IP-Raspberry-Pi>:8080/video_feed")
    print("Snapshot URL: http://<IP-Raspberry-Pi>:8080/capture_image")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8080, threaded=True)
