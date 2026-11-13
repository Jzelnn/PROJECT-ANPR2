#!/bin/bash
# ========================================================
# Skrip Menjalankan ANPR Parking System di Linux / Raspberry Pi
# ========================================================
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export USE_XNNPACK=0
export TORCH_DISABLE_XNNPACK=1
export TORCH_NUM_THREADS=1
export ANPR_TORCH_THREADS=1
export OPENCV_NUM_THREADS=1
export SKIP_WARMUP=1
export DISABLE_PADDLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Muat variabel dari file .env lokal jika tersedia (jangan di-commit ke git)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# ========================================================
# KONFIGURASI KEAMANAN (Dokumentasi Env Baru - Langkah 1)
# ========================================================
# - ANPR_API_TOKEN      : Token rahasia untuk autentikasi semua endpoint /api/* dan /ws/live
# - ANPR_SESSION_SECRET : Kunci rahasia HMAC untuk tanda tangan cookie sesi baca (12 jam)
# - ANPR_COOKIE_SECURE  : Setel ke 1 jika memakai reverse proxy HTTPS (Nginx/Caddy); 0 untuk HTTP LAN
#                         (Catatan: HTTP polos di LAN berisiko sniffing; gunakan reverse proxy HTTPS untuk produksi)
# - ANPR_AUTH_DISABLED  : Setel ke 1 HANYA jika ingin menonaktifkan autentikasi token pada jaringan terisolasi
# - ANPR_BIND_HOST      : Host bind server (default: 0.0.0.0; wajib set ANPR_API_TOKEN kecuali ANPR_AUTH_DISABLED=1)
# - ANPR_CCTV_USER      : Username kamera CCTV RTSP/ISAPI (default: admin)
# - ANPR_CCTV_PASSWORD  : Password kamera CCTV RTSP/ISAPI (dikosongkan di repo demi keamanan)
# - ANPR_CORS_ORIGINS   : Daftar origin CORS & WebSocket yang diizinkan (kosong = same-origin saja)
export ANPR_BIND_HOST="${ANPR_BIND_HOST:-0.0.0.0}"
export ANPR_COOKIE_SECURE="${ANPR_COOKIE_SECURE:-0}"
export ANPR_CCTV_STREAM_URL="${ANPR_CCTV_STREAM_URL:-rtsp://admin:Mddcoid*@192.168.110.64:554/Streaming/Channels/101}"
export ANPR_AUTO_START_CCTV="${ANPR_AUTO_START_CCTV:-1}"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Pengaturan Autentikasi:
if [ "${ANPR_AUTH_DISABLED:-0}" = "1" ]; then
    export ANPR_API_TOKEN=""
    rm -f .anpr_token 2>/dev/null || true
    echo "========================================================"
    echo "[AUTH] ANPR_AUTH_DISABLED=1 aktif: Server berjalan BEBAS TOKEN (No Auth)."
    echo "       Semua endpoint REST API & WebSocket dapat diakses tanpa token."
    echo "========================================================"
elif [ -z "$ANPR_API_TOKEN" ]; then
    ANPR_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))' 2>/dev/null || date +%s)"
    export ANPR_API_TOKEN
    # Simpan token ke file lokal dengan izin terbatas (chmod 600) agar aman
    echo "$ANPR_API_TOKEN" > .anpr_token
    chmod 600 .anpr_token 2>/dev/null || true
    echo "========================================================"
    echo "[SECURITY] ANPR_API_TOKEN otomatis dibuat dan disimpan ke .anpr_token (chmod 600)."
    echo "           Prefix token aktif: ${ANPR_API_TOKEN:0:6}***"
    echo "           Salin token dari file .env atau set ANPR_API_TOKEN milik Anda."
    echo "========================================================"
fi

python backend/app.py
