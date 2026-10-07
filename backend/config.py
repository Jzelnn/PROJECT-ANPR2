"""
Konfigurasi path proyek ANPR Parking.

Semua lokasi file (model, dashboard, output) diatur di sini, relatif terhadap root proyek,
sehingga backend bisa dijalankan dari folder mana pun. Setiap path model bisa dioverride
lewat environment variable (mis. ANPR_VEHICLE_MODEL=D:\\model\\best.pt).
"""
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Deteksi kendaraan Indonesia (Bus, Mobil, Motor, Pickup, Truck)
VEHICLE_MODEL_PATH = os.environ.get("ANPR_VEHICLE_MODEL", os.path.join(MODELS_DIR, "vehicle", "best.pt"))
# Deteksi plat nomor + petunjuk sub-tipe bodi (PRKING-ANPR, multi-kelas)
PLATE_MODEL_PATH = os.environ.get("ANPR_PLATE_MODEL", os.path.join(MODELS_DIR, "plate", "best.pt"))
# Detektor plat 1 kelas: fallback untuk plat gelap/kuning/militer bila detektor utama miss
PLATE_LEGACY_MODEL_PATH = os.environ.get("ANPR_PLATE_LEGACY_MODEL", os.path.join(MODELS_DIR, "plate", "legacy.pt"))
# Klasifikasi tipe bodi (Sedan, SUV, MPV, Hatchback, ...)
BODY_TYPE_MODEL_PATH = os.environ.get("ANPR_BODY_TYPE_MODEL", os.path.join(MODELS_DIR, "body_type", "best.pt"))
# Pengenalan karakter plat (0-9, A-Z)
CHARACTER_MODEL_PATH = os.environ.get("ANPR_CHARACTER_MODEL", os.path.join(MODELS_DIR, "character", "best.pt"))

# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
DASHBOARD_FILE = "parkir-anpr-dashboard.html"

# ------------------------------------------------------------
# Output runtime
# ------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")      # snapshot Entry History (URL: /captures/<file>)
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
DEBUG_CROPS_DIR = os.path.join(RESULTS_DIR, "debug_plates")      # crop plat debug (ANPR_DEBUG_CROPS=1)
TEMP_UPLOAD_PATH = os.path.join(RESULTS_DIR, "temp_upload.jpg")  # frame sementara dari RPi / CCTV snapshot

# ------------------------------------------------------------
# Data uji
# ------------------------------------------------------------
TEST_DIR = os.path.join(PROJECT_ROOT, "test")
TEST_VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
UPLOADED_VIDEO_PATH = os.path.join(TEST_VIDEOS_DIR, "uploaded_test_video.mp4")   # video dari fitur Upload Video

for _d in (SCREENSHOTS_DIR, RESULTS_DIR, LOGS_DIR, TEST_VIDEOS_DIR):
    os.makedirs(_d, exist_ok=True)


def require_file(path, what):
    """Gagal dengan pesan jelas bila file model tidak ada (daripada error ultralytics yang membingungkan)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} tidak ditemukan: {path}  (lihat README.md bagian 'Model')")
    return path
