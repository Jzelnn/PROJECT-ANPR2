# Jawaban Pertanyaan Terbuka Integrasi ALPR (V1)
**Dokumen Referensi Integrasi Sistem Parkir & Controller Gerbang**  
**Tanggal**: 13 November 2026  
**Status**: Final / Siap Diintegrasikan  

---

## 1. Bentuk Sambungan ALPR Web

### a. Port dan Protokol
Sistem ALPR baru berjalan pada **port 5001** dengan protokol **HTTP/REST** dan **WebSocket** (default bind ke `0.0.0.0:5001`).  
*(Catatan: Jika controller eksisting masih memanggil endpoint V2 lama seperti `/get_plate` pada port 5000, routing alias atau reverse proxy port 5000 -> 5001 dapat disediakan untuk kompatibilitas penuh).*

### b. Endpoint Utama Integrasi
| Method | Endpoint | Fungsi |
|---|---|---|
| `POST` | `/api/stream/capture` | Trigger instan dari sensor loop/tombol dispenser untuk menjepret frame CCTV dan menjalankan ANPR (< 1 detik). |
| `POST` | `/api/detect` | Mengirim file gambar (`multipart/form-data` dengan field `image`) untuk dianalisis plat & jenis kendaraannya. |
| `GET` | `/api/detect_current` | Mengambil data kendaraan yang sedang terdeteksi di area deteksi (ROI) saat ini dari memory RAM (0ms latency). |
| `GET` | `/api/history` | Mengambil riwayat kendaraan masuk/terkonfirmasi terbaru beserta link foto snapshot. |
| `GET` | `/api/live_stream` | Stream video MJPEG real-time untuk ditampilkan di UI pos satpam. |
| `WS` | `/ws/live` | WebSocket event stream yang mem-broadcast event secara otomatis begitu plat nomor terkonfirmasi (`CONFIRMED`). |

### c. Alamat Dashboard Web
- **URL Dashboard**: `http://<IP_ALPR>:5001/dashboard` (atau root `http://<IP_ALPR>:5001/`)
- Menampilkan live preview kamera, bounding box kendaraan, pembacaan plat nomor, status sistem, tombol ubah area ROI, dan tabel riwayat transaksi.

### d. Contoh Respons JSON (`POST /api/stream/capture` atau `POST /api/detect`)
```json
{
  "status": "success",
  "detection_time_sec": 0.586,
  "detections": [
    {
      "track_id": 1,
      "license_plate": "B 1489 VEH",
      "plate_confidence": 0.85,
      "vehicle_type": "car",
      "vehicle_confidence": 0.728,
      "body_style": "SUV",
      "body_style_confidence": 0.82,
      "status": "CONFIRMED",
      "inside_interest_area": true,
      "is_newly_confirmed": true,
      "bbox": [174, 32, 515, 410],
      "plate_bbox": [280, 290, 395, 340]
    }
  ],
  "image_url": "/captures/20261113_100927_B1489VEH.jpg"
}
```

### e. Cara Akses & Autentikasi
- **Jenis Autentikasi**: **Bearer Token** / **Header API Token**.
  - Header request: `Authorization: Bearer <TOKEN>` atau `X-API-Token: <TOKEN>`.
  - Browser dashboard menggunakan **Session Cookie** terenkripsi HMAC (`anpr_session`).
- **Opsi Jaringan Lokal (Isolasi)**: Jika berada pada jaringan LAN tertutup/terisolasi dan controller tidak mendukung header token, autentikasi dapat dinonaktifkan dengan mengatur environment `ANPR_AUTH_DISABLED=1`.

---

## 2. Akses dari Laptop / Jaringan Lapangan

- **Dapat dijangkau dari jaringan laptop**: **YA**.
- **Alamat IP ALPR saat ini**: **`192.168.110.7`** (Port **`5001`**).
- **Format URL Base**: `http://192.168.110.7:5001`
- Anda dapat melakukan ping langsung ke `192.168.110.7` atau membuka `http://192.168.110.7:5001/api/health` untuk uji konektivitas.

---

## 3. Pemanfaatan Hasil ALPR

Hasil ALPR mencakup seluruh kebutuhan operasional gerbang:

- **(a) Pencatatan**: **YA**, setiap plat nomor yang terkonfirmasi otomatis dicatat ke riwayat parkir (`output/parking_history.json` dan memory RAM buffer), lengkap dengan tanggal, jam, plat, tipe kendaraan, dan file snapshot.
- **(b) Ditampilkan di Layar Gerbang**: **YA**, bisa ditampilkan langsung lewat Web Dashboard pos kasir, atau dikirimkan datanya ke Running Text LED / Display Pos melalui polling REST API atau push WebSocket `/ws/live`.
- **(c) Pencocokan Plat Masuk vs Keluar (*Fuzzy Matching*)**: **YA / DIDUKUNG**. Sistem ALPR memiliki modul kalkulasi kemiripan plat (*fuzzy similarity*) dengan normalisasi karakter ambigu (contoh `O` dan `0`, `I` dan `1`, `Z` dan `2`). Toleransi kemiripan (misal $\ge 70\%$) siap diterapkan di pintu keluar untuk memvalidasi kendaraan.
- **(d) Penentuan Tarif Berdasarkan Jenis Kendaraan**: **YA**. ALPR menghasilkan klasifikasi kendaraan:
  - `car` (Mobil)
  - `motorcycle` (Sepeda Motor)
  - `bus` (Bus)
  - `truck` (Truk)
  - Plus sub-tipe bodi (*Sedan, SUV, MPV, Hatchback, Pickup*).  
  Sistem billing/pos kasir dapat langsung menggunakan field `vehicle_type` untuk menghitung tarif parkir tanpa ketergantungan pada sensor loop multi-kategori.

---

## 4. Foto Kendaraan & Kebijakan Masa Simpan (*Retention Policy*)

1. **Penyimpanan Foto**:
   - **YA**, foto kendaraan otomatis disimpan setiap kali kendaraan terkonfirmasi.
   - Foto yang disimpan adalah frame beranotasi beresolusi jelas (menampilkan mobil dan kotak plat nomor).
   - Lokasi penyimpanan: `output/screenshots/<timestamp>_<plat>.jpg`.
   - File dapat diakses langsung oleh sistem kasir via URL: `http://192.168.110.7:5001/captures/<filename>`.
2. **Masa Simpan Data (*Data Retention*)**:
   - Ditetapkan standar **180 hari** (mengikuti regulasi perlindungan data dan audit sistem parkir lama).
   - File snapshot dan log yang melewati batas 180 hari akan di-purge secara otomatis untuk efisiensi penyimpanan disk Raspberry Pi.
