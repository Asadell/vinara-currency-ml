# 🚀 GUIDO / VINARA: AI Handover & Execution Guide

Dokumen ini dibuat secara khusus untuk AI Assistant / Agent pengembang agar dapat **langsung mengeksekusi, memantau, mendebug, dan mengintegrasikan** seluruh ekosistem **Rupiah Vision ML Training (Backend)** dan **GUIDIO Flutter Mobile App**.

---

## 🛠️ 1. Struktur Repositori Project

Workspace ini terdiri dari beberapa repositori Git utama:
1. **Mobile App (Flutter)**: `/home/asadel/kuliah/lomba/smstr6/guido/project/guidio_app`
   - Framework: Flutter / Dart (Branch: `develop`)
   - TFLite Model Location: `assets/models/rupiah_classifier_int8.tflite` atau `rupiah_classifier_fp16.tflite`
   - Test File: `test/model_inference_test.dart`

2. **Rupiah ML Training (Backend)**: `/home/asadel/kuliah/lomba/smstr6/guido/new_training/rupiah_vision_revised`
   - Repository Remote: `git@github.com:Asadell/vinara-currency-ml.git` (Branch: `develop`)
   - Framework: TensorFlow 2.21 + Keras + Albumentations + AI-Edge LiteRT

3. **Gambar Fixture Test Nyata**: `/home/asadel/kuliah/lomba/smstr6/guido/test/rupiah/`
   - Berisi 8 gambar uang Rupiah asli (`5_ribu_a.png`, `10_ribu_a.png`, `20_ribu_a.png`, `50_ribu_a.png`, dst).

---

## 🖥️ 2. VPS GPU Vast.ai Environment & SSH Configuration

Sistem ML Backend dilatih di Remote VPS GPU (Vast.ai) dengan spesifikasi:
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **RAM**: 125GB System RAM (`TMPDIR=/dev/shm` digunakan untuk RAM Disk I/O)
- **SSH Connection**:
  ```bash
  ssh -i ~/.ssh/id_vastai -p 37281 root@1.193.137.175
  ```
- **Repo VPS Location**: `/root/vinara-currency-ml`
- **Dataset Roboflow Auto-Location**: `/root/datasets/rupiah-detection/` (3 dataset: `rf-rupiah-skripsi`, `rf-rupiah-detector`, `rf-money-detection-valid`)
- **Folder Test Fixture di VPS**: `/root/test_rupiah_vps/`

---

## ⚡ 3. Cara Menjalankan ML Pipeline di VPS (Backend Training)

### A. Perintah Utama Eksekusi Remote (Nohup Background)
Training dijalankan menggunakan runner script `/root/run_rupiah_train.sh` yang diputus dari hup terminal lokal:

```bash
TERM=xterm-256color ssh -i ~/.ssh/id_vastai -p 37281 root@1.193.137.175 -o StrictHostKeyChecking=no "
cd /root/vinara-currency-ml
git pull origin develop
nohup /root/run_rupiah_train.sh > /root/rupiah_train.log 2>&1 &
echo 'RUNNER_PID='\$!
"
```

### B. Urutan Alur Script Runner (`/root/run_rupiah_train.sh`):
1. **Step 1 (Deduplikasi & Crop BBox)**: `python3 scripts/00_merge_and_crop.py`
   - Secara otomatis mendeteksi 3 dataset Roboflow di `~/datasets/rupiah-detection/rf-*`.
   - Menghasilkan 6.494 foto crop unik seimbang (7 kelas: 1000, 2000, 5000, 10000, 20000, 50000, 100000).
2. **Step 2 (Training MobileNetV2)**: `python3 scripts/01_train.py --batch-size 64`
   - Auto-detect GPU (`NVIDIA GeForce RTX 4090`).
   - Auto-enable **FP16 Mixed Precision** (`mixed_float16`).
   - Simulasi Procedural Scene Framing & Albumentations (termasuk rotasi, lipatan, dan bayangan).
   - **Tahap 1 (Head Training)**: 12 Epochs (`trainable=False` backbone).
   - **Tahap 2 (Fine-Tuning)**: 45 Epochs (`trainable=True` seluruh backbone dengan learning rate kecil).
   - Auto export TFLite (`rupiah_classifier_int8.tflite` & `rupiah_classifier_fp16.tflite`).
3. **Step 3 (Evaluasi Fixture Test Nyata)**: `python3 /root/eval_test_rupiah.py`
   - Menguji 8 foto nyata di `/root/test_rupiah_vps/` dan mencetak persentase akurasi & confidence score.

### C. Cara Memantau Progress Log Training
```bash
TERM=xterm-256color ssh -i ~/.ssh/id_vastai -p 37281 root@1.193.137.175 -o StrictHostKeyChecking=no "
cat /root/rupiah_train.log | tail -40
"
```

---

## 📱 4. Cara Integrasi Model TFLite Baru ke Aplikasi Flutter Mobile

Setelah training di VPS selesai 100%:

### A. Download Model TFLite Terbaru dari VPS ke Lokal:
```bash
mkdir -p /home/asadel/kuliah/lomba/smstr6/guido/downloaded_models
scp -P 37281 -i ~/.ssh/id_vastai root@1.193.137.175:/root/vinara-currency-ml/models/*.tflite /home/asadel/kuliah/lomba/smstr6/guido/downloaded_models/
```

### B. Salin Model ke Folder Asset Flutter App:
```bash
cp /home/asadel/kuliah/lomba/smstr6/guido/downloaded_models/rupiah_classifier_int8.tflite /home/asadel/kuliah/lomba/smstr6/guido/project/guidio_app/assets/models/
cp /home/asadel/kuliah/lomba/smstr6/guido/downloaded_models/rupiah_classifier_fp16.tflite /home/asadel/kuliah/lomba/smstr6/guido/project/guidio_app/assets/models/
```

### C. Jalankan Unit/Integration Test Flutter:
```bash
cd /home/asadel/kuliah/lomba/smstr6/guido/project/guidio_app
flutter test test/model_inference_test.dart
```

---

## 💡 5. Catatan Penting untuk AI Agent
- **Kuis & Git Commit**: Selalu gunakan Conventional Commits (`feat:`, `fix:`, `docs:`, `perf:`).
- **Eksekusi Training**: JANGAN PERNAH menjalankan script training berat di komputer lokal. Selalu eksekusi di VPS GPU Vast.ai (`1.193.137.175:37281`).
- **Mixed Precision**: `mixed_float16` sudah terkonfigurasi secara otomatis di `01_train.py` bila GPU terdeteksi.
