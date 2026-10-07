"""
Script Streaming Kamera untuk Raspberry Pi
Jalankan file ini di Raspberry Pi Anda:
    python3 rpi_camera_streamer.py

Video akan dipancarkan di:
    http://<IP-Raspberry-Pi>:8080/video_feed
"""

import cv2
from flask import Flask, Response

app = Flask(__name__)

# Gunakan index 0 untuk USB Webcam atau Pi Camera (V4L2)
# Jika memakai Picamera2 di RPi 4/5, Anda juga bisa menggantinya dengan modul picamera2.
camera = cv2.VideoCapture(0)

# Set resolusi streaming optimal untuk transmisi jaringan nirkabel (WiFi)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

def generate_mjpeg():
    """Mengambil frame dari kamera dan membroadcast dalam format MJPEG stream."""
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        # Kompres ke JPEG kualitas 80% (ringan dan jernih)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return "<h3>Kamera Raspberry Pi ANPR Aktif!</h3><p>Stream URL: <a href='/video_feed'>/video_feed</a></p>"

@app.route('/video_feed')
def video_feed():
    """Endpoint HTTP MJPEG Stream untuk web dashboard ANPR."""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 60)
    print("🎥 Raspberry Pi Camera Streamer Aktif!")
    print("Hubungkan dari Laptop di: http://<IP-Raspberry-Pi>:8080/video_feed")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8080, threaded=True)
