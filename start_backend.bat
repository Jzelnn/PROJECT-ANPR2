@echo off
title ANPR Parking System Server - Port 5001
color 0A
echo ========================================================
echo    MENJALANKAN SERVER ANPR PARKIR (AI & CCTV)...
echo ========================================================
echo.
echo Server sedang memuat model AI (YOLO + Body Type + OCR)...
echo Browser akan otomatis terbuka ke http://localhost:5001
echo.
echo Tekan CTRL + C di jendela ini jika ingin mematikan server.
echo ========================================================
echo.

cd /d "%~dp0"

:: ========================================================
:: KONFIGURASI KEAMANAN (Dokumentasi Env Baru - Langkah 1)
:: ========================================================
:: - ANPR_API_TOKEN      : Token rahasia untuk autentikasi endpoint /api/* dan /ws/live
:: - ANPR_SESSION_SECRET : Kunci rahasia HMAC untuk tanda tangan cookie sesi baca (12 jam)
:: - ANPR_COOKIE_SECURE  : Setel ke 1 jika memakai reverse proxy HTTPS (Nginx/Caddy); 0 untuk HTTP LAN
::                         (Catatan: HTTP polos di LAN berisiko sniffing; gunakan reverse proxy HTTPS untuk produksi)
:: - ANPR_AUTH_DISABLED  : Setel ke 1 HANYA jika ingin menonaktifkan autentikasi token di jaringan lokal terisolasi
:: - ANPR_BIND_HOST      : Host bind server (default: 0.0.0.0)
:: - ANPR_CCTV_USER      : Username kamera CCTV RTSP/ISAPI
:: - ANPR_CCTV_PASSWORD  : Password kamera CCTV RTSP/ISAPI (jangan hardcode di kode!)
:: - ANPR_CORS_ORIGINS   : Daftar origin CORS & WebSocket yang diizinkan (kosong = same-origin saja)
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%~A:~0,1%"=="#" set "%%A=%%B"
    )
)

if "%ANPR_BIND_HOST%"=="" set ANPR_BIND_HOST=0.0.0.0
if "%ANPR_COOKIE_SECURE%"=="" set ANPR_COOKIE_SECURE=0

set PYTHON_CMD=python
if exist "C:\Users\Lenovo\AppData\Local\Programs\Python\Python39\python.exe" (
    set "PYTHON_CMD=C:\Users\Lenovo\AppData\Local\Programs\Python\Python39\python.exe"
)

if "%ANPR_API_TOKEN%"=="" if not "%ANPR_AUTH_DISABLED%"=="1" (
    for /f %%i in ('"%PYTHON_CMD%" -c "import secrets; print(secrets.token_urlsafe(24))"') do set "ANPR_API_TOKEN=%%i"
    echo [SECURITY] ANPR_API_TOKEN telah disiapkan otomatis untuk sesi ini.
    echo            Atur ANPR_API_TOKEN di file .env untuk token tetap.
)

:: Buka browser otomatis setelah 3 detik
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5001"

"%PYTHON_CMD%" backend\app.py

pause
