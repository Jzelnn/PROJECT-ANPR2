# LAPORAN PROGRES HARIAN PROYEK ANPR (AUTOMATIC NUMBER PLATE RECOGNITION)
**Tanggal**: 9 November 2026  
**Target Hardware**: Raspberry Pi 4 Model B (Quad-core Cortex-A72 ARMv8-A 64-bit)  
**Status Sistem**: 🟢 **BERHASIL AKTIF (Server Live di Port 5001)**

---

## 1. Ringkasan Eksekutif (Executive Summary)
Hari ini fokus utama adalah melakukan **deployment dan eksekusi engine AI ANPR langsung di dalam perangkat fisik Raspberry Pi 4** (*edge middleware*). Sistem sebelumnya mengalami kendala kompatibilitas biner CPU (`Illegal instruction / SIGILL`) karena library pre-compiled AI modern mengasumsikan instruksi server ARMv8.2 (FP16/BFLOAT16). 

Melalui serangkaian diagnosa bertahap, arsitektur inferensi telah berhasil diisolasi dan dialihkan ke **Pure PyTorch Forward Pass + Torchvision NMS**, sehingga seluruh model AI dapat berjalan mulus di CPU Cortex-A72. Server Flask dan WebSocket telah sukses mengudara di port **`5001`** dan melayani dashboard pemantauan lalu lintas parkir.

---

## 2. Pencapaian Hari Ini (Today's Key Achievements)

### A. Konfigurasi Lingkungan 64-bit (aarch64)
- Berhasil menyiapkan *virtual environment* Python 64-bit di `/home/parkour/PROJECT ANPR2/venv`.
- Dependensi inti terpasang lengkap: `torch`, `torchvision`, `ultralytics`, `flask`, `flask-cors`, `flask-sock`, `opencv-python-headless`, `sqlite3`, dan `numpy`.

### B. Penyelesaian Isu Kritis Arsitektur CPU (`Illegal instruction`)
- **Akar Masalah**: Fungsi bawaan `model.predict()` pada Ultralytics mengaktifkan akselerasi otomatis `autocast / AMP` yang mengeksekusi instruksi vektor ARMv8.2 `BFLOAT16`, serta PaddleOCR yang memicu crash biner di ARMv8.0.
- **Solusi yang Diterapkan & Tervalidasi**:
  1. **Mode Cepat Pure PyTorch (`DISABLE_PADDLE=1`)**: Menonaktifkan runtime PaddleOCR dan menggantikannya dengan modul deteksi karakter plat berbasis PyTorch YOLO Character Model (`models/character/best.pt`) yang jauh lebih cepat (~35ms).
  2. **Isolasi Threading & Vector Flags**: Memasang `OPENBLAS_CORETYPE=ARMV8`, `USE_XNNPACK=0`, `TORCH_DISABLE_XNNPACK=1`, serta mengunci eksekusi inferensi ke *single-thread* (`torch.set_num_threads(1)`) untuk mencegah tabrakan *atomic lock*.
  3. **Implementasi Pure ARM Predictor (`pure_arm_yolo_predict`)**: Mem-bypass wrapper bawaan Ultralytics dan langsung mengeksekusi *forward pass* matematika tensor PyTorch murni dipadukan dengan `torchvision.ops.nms`.
- **Hasil Uji Diagnosa**:
  - `Vehicle Model (best.pt)`: **100% SUKSES**
  - `Plate Best Model (best.pt)`: **100% SUKSES**
  - `Plate Legacy Model (legacy.pt)`: **100% SUKSES**
  - `Character Recognition Model (best.pt)`: **100% SUKSES**
  - `Body Type Classification Model (best.pt)`: **100% SUKSES**

### C. Peluncuran Layanan Dashboard Real-Time
- Backend server ANPR berhasil dinyalakan melalui skrip `start_backend.sh`.
- Dashboard web dan WebSocket berhasil mengikat port jaringan lokal: **`http://192.168.110.7:5001`**.
- Client berhasil mengakses `/`, memuat riwayat parkir via `/api/history`, dan terhubung via WebSocket secara stabil.

---

## 3. Status Terkini (Current State)
- **Repositori Git**: Cabang `main` telah sinkron hingga commit **`54bfd35`** (*feat(engine): hook PureArmYoloPredictor into YOLO.predict to eliminate Ultralytics predict SIGILL on Cortex-A72*).
- **Keamanan Data**: Semua source code, skrip otomatisasi, konfigurasi model, dan struktur database tersimpan aman di GitHub.
- **Kondisi Perangkat**: Raspberry Pi siap untuk dihubungkan kembali via SSH (`ssh parkour@192.168.110.7`) pasca-reboot laptop.

---

## 4. Rencana Kerja Besok (Tomorrow's Action Plan)

| No | Agenda Kerja | Deskripsi & Target Teknis |
| :---: | :--- | :--- |
| **1** | **Verifikasi End-to-End Upload Foto** | Menjalankan pengujian tombol "Upload Foto" di dashboard dengan citra kendaraan asli Indonesia, memverifikasi pembacaan plat nomor, tipe bodi, serta pencatatan otomatis snapshot ke `parking.db`. |
| **2** | **Integrasi RTSP / Local Camera Feed** | Menghubungkan feed video RTSP IP Camera gate masuk atau webcam USB ke thread background pemrosesan `app.py`. |
| **3** | **Tuning Latensi & Manajemen Suhu Pi 4** | Mengukur rata-rata latensi deteksi per frame pada resolusi 384x384 (target $\le$ 300ms/frame), mengatur parameter `ANPR_FRAME_SKIP=2`, dan memantau suhu CPU agar tetap stabil di bawah 65°C. |
| **4** | **Konfigurasi Auto-Start Service (`systemd`)** | Membuat unit file service Linux `/etc/systemd/system/anpr.service` agar sistem ANPR otomatis running di latar belakang (*daemon*) saat Raspberry Pi pertama kali dinyalakan tanpa perlu SSH manual. |
| **5** | **Uji Validasi Plat Dinas & Skenario Gate Out** | Menguji akurasi aturan regex plat nomor sipil, plat dinas (TNI/Polri/RI), serta mekanisme pencocokan durasi parkir & tarif saat kendaraan keluar. |

---

## 5. Ringkasan Perintah Penting (Quick Reference)

Untuk memulai sistem kembali besok pagi:
```bash
# 1. Masuk ke Raspberry Pi
ssh parkour@192.168.110.7

# 2. Masuk ke direktori & update kode
cd ~/PROJECT\ ANPR2
git reset --hard
git pull

# 3. Jalankan backend ANPR
bash start_backend.sh

# 4. Buka di browser
# http://192.168.110.7:5001
```
