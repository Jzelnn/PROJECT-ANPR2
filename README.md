# ANPR Parking

Sistem pengenalan plat nomor (ANPR/ALPR) untuk parkir: deteksi kendaraan (YOLO), deteksi plat, OCR plat
(PaddleOCR + model karakter), klasifikasi tipe bodi, dan dashboard web real-time dengan Entry History.

## Struktur folder

```
ANPR-Parking/
├── backend/
│   ├── app.py                 # server Flask + pipeline ANPR (port 5001)
│   ├── config.py              # semua path (model, frontend, output)
│   ├── requirements.txt
│   └── rpi/                   # skrip pendukung Raspberry Pi (kamera & monitor)
├── frontend/
│   └── parkir-anpr-dashboard.html
├── models/
│   ├── vehicle/best.pt        # deteksi kendaraan Indonesia (Bus, Mobil, Motor, Pickup, Truck)
│   ├── plate/best.pt          # deteksi plat + petunjuk sub-tipe bodi (PRKING-ANPR, 19 kelas)
│   ├── plate/legacy.pt        # detektor plat 1 kelas, fallback untuk plat gelap/kuning/militer
│   ├── body_type/best.pt      # klasifikasi tipe bodi (Sedan, SUV, MPV, Hatchback, ...)
│   ├── character/best.pt      # pengenalan karakter plat (0-9, A-Z)
│   └── _archive/              # hasil training lain yang tidak dipakai (tidak dimuat)
├── datasets/                  # dataset training per model
│   ├── vehicle/  license_plate/  body_type/  character/
├── test/
│   ├── images/                # foto uji (kendaraan, crop plat, frame CCTV)
│   └── videos/                # rekaman CCTV / video uji
├── output/
│   ├── screenshots/           # snapshot Entry History (disajikan di /captures/<file>)
│   ├── results/               # hasil & file sementara (debug crop plat, temp upload)
│   └── logs/
├── docs/srs_summary.txt
├── start_backend.bat          # jalankan server + buka dashboard
└── install_requirements.bat   # pasang dependensi Python
```

## Menjalankan

1. Pasang dependensi (sekali): jalankan `install_requirements.bat`
   (atau `pip install -r backend/requirements.txt`).
2. Jalankan server: `start_backend.bat` (atau `python backend/app.py` dari root proyek).
3. Buka dashboard di http://localhost:5001.

Server membaca semua path dari `backend/config.py`, jadi bisa dijalankan dari folder mana pun.

## Model

Backend memuat **lima** file model. Bila salah satu tidak ada, server berhenti dengan pesan yang
menyebutkan path yang hilang.

| Model | File | Override (environment variable) |
|---|---|---|
| Kendaraan | `models/vehicle/best.pt` | `ANPR_VEHICLE_MODEL` |
| Plat nomor | `models/plate/best.pt` | `ANPR_PLATE_MODEL` |
| Plat (fallback) | `models/plate/legacy.pt` | `ANPR_PLATE_LEGACY_MODEL` |
| Tipe bodi | `models/body_type/best.pt` | `ANPR_BODY_TYPE_MODEL` |
| Karakter plat | `models/character/best.pt` | `ANPR_CHARACTER_MODEL` |

Untuk mengganti model dengan hasil training baru, timpa `best.pt` di folder kategorinya
(atau arahkan environment variable ke file lain) lalu restart server.

Isi `models/_archive/`:

| File | Isi |
|---|---|
| `vehicle_old_4class.pt` | model kendaraan lama (car, motorcycle, bus, truck) |
| `vehicle_indo_last.pt` | checkpoint terakhir training kendaraan Indonesia |
| `plate_prking_last.pt` | checkpoint terakhir training plat PRKING-ANPR |
| `plate_platnomor1_best.pt` / `_last.pt` | training plat "Plat-Nomor-1" (6 kelas) |
| `character_v4_best.pt` / `_last.pt` | training karakter dataset v4 (berbeda dari model aktif) |
| `character_v2_best.pt` / `_last.pt` | training karakter dataset v2 |
| `yolo11n_coco.pt` | YOLO11n bawaan (COCO 80 kelas) |

Bobot model tidak di-commit ke git (lihat `.gitignore`), kecuali yang sudah dilacak sebelumnya.

## API utama

| Endpoint | Fungsi |
|---|---|
| `GET /` | dashboard |
| `POST /api/detect` | deteksi pada satu gambar (`is_stream=true` untuk frame video/webcam dari browser) |
| `POST /api/stream/start`, `/api/stream/stop` | stream CCTV/RTSP atau file video lokal di server |
| `POST /api/upload_video` | unggah video uji lalu putar sebagai stream |
| `GET /api/history`, `/api/export` | Entry History |
| `GET/POST /api/config/interest_area` | Detection Area (ROI) |
| `WS /ws/live` | pembaruan deteksi real-time |
