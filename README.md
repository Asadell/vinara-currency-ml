# Rupiah Vision - Training Pipeline (Revisi)

Pipeline klasifikasi uang kertas Rupiah (7 kelas) berbasis MobileNetV2 -> TFLite,
direvisi total supaya **robust di uang lecek, terlipat, dicoret, dan cahaya buruk**,
bukan cuma akurat di dataset bersih.

---

## Baca ini dulu: kenapa akurasi 99.52% kamu kemungkinan besar palsu

Versi lama `00_merge_and_crop.py` mengumpulkan semua crop lalu:

```python
random.shuffle(all_crops)
n_test = int(n * test_split)
...
```

Ini split **per crop**, bukan per foto sumber. Dua konsekuensinya:

1. **Satu foto berisi beberapa lembar uang.** Crop A masuk train, crop B dari
   foto yang sama masuk test. Background, pencahayaan, sudut, dan sensor-nya
   identik. Model tinggal menghafal konteksnya.

2. **Dataset Roboflow banyak yang dari frame video.** Foto ke-31 dan ke-32
   praktis sama persis. Satu masuk train, satu masuk test.

Efeknya, test set kamu bukan mengukur generalisasi, tapi mengukur hafalan.
Makanya angkanya 99.52% tapi begitu ketemu uang lecek di warung, model bingung.
Augmentasi sekuat apa pun tidak akan memperbaiki ini kalau evaluasinya masih bocor,
karena kamu tidak punya cara jujur untuk tahu apakah perbaikanmu berhasil.

**Yang berubah:** split sekarang dilakukan per **grup** (satu foto sumber = satu
grup), plus deduplikasi perceptual hash, plus verifikasi kebocoran otomatis di
`00b_preflight_check.py`.

Siap-siap: setelah regenerate dataset, angka test accuracy kamu **akan turun**,
mungkin ke 92-96%. Itu bukan kemunduran. Itu angka pertama yang jujur, dan baru
dari situ perbaikan bisa diukur.

---

## Struktur

```
rupiah_vision_revised/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── common.py     # kelas, letterbox, pHash, statistik citra
│   ├── augment.py    # augmentasi uang lecek (Albumentations + custom)
│   └── data.py       # tf.data pipeline, balanced sampling, MixUp/CutMix
└── scripts/
    ├── 00_merge_and_crop.py       # merge + crop + dedup + group-aware split
    ├── 00b_preflight_check.py     # verifikasi dataset (termasuk cek kebocoran)
    ├── 01_train.py                # training MobileNetV2
    ├── 02_export_tflite.py        # ekspor FP32/FP16/INT8
    └── 03_calibrate_threshold.py  # kalibrasi + ambang tolak
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`albumentations` wajib. Tanpa itu pipeline jatuh ke augmentasi TF bawaan yang
jauh lebih lemah (tanpa elastic, perspective, simulasi lipatan, simulasi kusut).

---

## Alur lengkap

---

## Dataset Sumber (16 Dataset Roboflow Megamix — Total 1.09 GB)

Pipeline ini secara otomatis mengunduh, menggabungkan, memotong bounding box (crop), dan menstandardisasi **16 dataset Roboflow** langsung di VPS GPU untuk mencakup variasi emisi baru (2022), emisi lama (2016), kondisi lecek, pencahayaan minim, serta sudut pengambilan gambar yang beragam:

### Tabel Rincian Dataset Sumber di VPS GPU

| No | Dataset Roboflow | Folder (`~/datasets/rupiah-detection/`) | Ukuran File | Jumlah Gambar Sumber |
|:---:|---|---|:---:|:---:|
| 1 | `skripsi-3kth2/deteksi-mata-uang-rupiah-nerog (v2)` | `rf-rupiah-skripsi` | **30.65 MB** | 608 |
| 2 | `rupiah-detector/rupiah-detector-qzmb7 (v2)` | `rf-rupiah-detector` | **96.90 MB** | 1,941 |
| 3 | `workspace1-u35mt/money-detection-valid (v4)` | `rf-money-detection-valid` | **236.83 MB** | 9,101 |
| 4 | `moneysaver-yolo/deteksi-uang-s0pfe` | `rf-moneysaver-yolo` | **40.66 MB** | 914 |
| 5 | `skripsi-swuyl/detetksi-keaslian-uang` | `rf-skripsi-swuyl` | *(Auto-fallback v1)* | *(merged)* |
| 6 | `yoloai-3iuvz/uang_deteksi` | `rf-yoloai-3iuvz` | **44.56 MB** | 3,360 |
| 7 | `amndan/uangbaru2022` | `rf-amndan` | **4.33 MB** | 247 |
| 8 | `jemy07s-workspace/tubes-psi-ydfzm` | `rf-jemy07s` | **86.13 MB** | 2,570 |
| 9 | `adelias-workspace/modeluangv2-t1lk9` | `rf-adelias` | **223.81 MB** | 4,954 |
| 10 | `zannho/rupiah-detection-o3agm` | `rf-zannho` | **36.12 MB** | 1,076 |
| 11 | `muhammad-aidil-wlsfe/cnnyolo-hhphe` | `rf-muhammad-aidil` | **83.25 MB** | 2,740 |
| 12 | `cahyadin/money_detection-pnnd7` | `rf-cahyadin` | **34.52 MB** | *(empty)* |
| 13 | `4ia17ottos-workspace/moneydetection-uetq8` | `rf-4ia17ottos-uetq8` | **197.86 MB** | 3,426 |
| 14 | `tes-nms6d/uang-kertas-2022-dan-logam-2016` | `rf-tes-nms6d` | *(Auto-fallback v1)* | *(merged)* |
| 15 | `4ia17ottos-workspace/uang_baru` | `rf-4ia17ottos-uang-baru` | *(Auto-fallback v1)* | *(merged)* |
| 16 | `project-binus/rupiah-detection-d1vbz` | `rf-project-binus` | *(Auto-fallback v1)* | *(merged)* |
| **TOTAL** | **16 Dataset Combined Megamix** | `~/datasets/rupiah-detection/` | **1.09 GB** *(1,115.61 MB)* | **30,937 Gambar Sumber** |

---

### Hasil Akhir Ekstraksi & Split Classification Dataset (`data/classification`)

Setiap bounding box dipotong (crop) dengan padding adaptif, dideplikasi menggunakan pHash 64-bit anti-leakage, dan dibagikan secara group-stratified:

- **Total Gambar Sumber**: 30,937 foto
- **Total Grup Foto Unik**: 18,796 foto (mencegah kebocoran video frame / burst photo)
- **Crop BBox Kualitas Tinggi Dipertahankan**: **22,147 crop**
- **Duplikat Dibuang (pHash)**: 17,831 crop
- **Dibuang (Blur / Kekecilan / Koin)**: 3,331 crop
- **Imbalance Ratio**: **1.34x** (Sangat Seimbang!)

| Nominal | Train (75%) | Val (15%) | Test (10%) | Total Crop |
|:---:|:---:|:---:|:---:|:---:|
| **Rp 1,000** | 2,448 | 486 | 327 | **3,261** |
| **Rp 2,000** | 2,072 | 421 | 266 | **2,759** |
| **Rp 5,000** | 2,769 | 552 | 368 | **3,689** |
| **Rp 10,000** | 2,459 | 493 | 343 | **3,295** |
| **Rp 20,000** | 2,248 | 446 | 298 | **2,992** |
| **Rp 50,000** | 2,380 | 459 | 318 | **3,157** |
| **Rp 100,000** | 2,265 | 437 | 292 | **2,994** |
| **TOTAL** | **16,641** | **3,294** | **2,212** | **22,147 Crop** |

---

### Normalisasi Label Otomatis (`normalize_class_name`)
Script `00_merge_and_crop.py` secara otomatis membaca `data.yaml` tiap dataset dan memetakan semua alias/sinonim label ke **7 nominal standar**:

| Target Standar | Sinonim / Alias yang Dinormalisasi |
|---|---|
| `1000` | `1k`, `1ribu`, `1rb`, `1.000`, `seribu`, `1000`, `Rp 1.000` |
| `2000` | `2k`, `2ribu`, `2rb`, `2.000`, `dua ribu`, `2000`, `Rp 2.000` |
| `5000` | `5k`, `5ribu`, `5rb`, `5.000`, `lima ribu`, `5000`, `Rp 5.000` |
| `10000` | `10k`, `10ribu`, `10rb`, `10.000`, `sepuluh ribu`, `10000`, `Rp 10.000` |
| `20000` | `20k`, `20ribu`, `20rb`, `20.000`, `dua puluh ribu`, `20000`, `Rp 20.000` |
| `50000` | `50k`, `50ribu`, `50rb`, `50.000`, `lima puluh ribu`, `50000`, `Rp 50.000` |
| `100000` | `100k`, `100ribu`, `100rb`, `100.000`, `seratus ribu`, `100000`, `Rp 100.000` |

*Catatan: Label koin/logam (`500 koin`, `1000_koin`), mata uang asing, atau objek luar secara otomatis dikesampingkan agar fokus pada 7 nominal uang kertas Rupiah.*

---

## Setup & Download di Remote GPU (Vast.ai)

> [!IMPORTANT]
> **Selalu jalankan download dan training di Remote GPU (Vast.ai), BUKAN di laptop lokal.** Laptop lokal hanya untuk mengedit code dan dokumentasi.

### Cara Download, Crop, & Train di VPS GPU:

```bash
# 1. SSH ke GPU
ssh -i ~/.ssh/id_vastai -p 37281 root@1.193.137.175

# 2. Masuk ke folder repo & pull update code terbaru
cd /root/vinara-currency-ml
git pull origin develop

# 3. Jalankan script download (dengan ROBOFLOW_API_KEY)
export ROBOFLOW_API_KEY="FOdZd5fsYRPdf0n5SEEX"
python3 scripts/download_rupiah.py

# 4. Merge, crop, dedup, dan split dataset
python3 scripts/00_merge_and_crop.py --clean

# 5. Preflight check
python3 scripts/00b_preflight_check.py --data data/classification --deep

# 6. Training Model Rupiah High-Speed GPU
bash scripts/run_vast.sh
```

---

## Alur lengkap

### Step 0 - Merge, crop, dedup, split

```bash
python scripts/00_merge_and_crop.py \
    --output data/classification \
    --val-split 0.15 --test-split 0.10 \
    --dedup-threshold 4 \
    --min-blur 25 \
    --verify-leakage \
    --clean
```

Format `--extra-dirs`: folder yang isinya sudah tersusun per nominal.

```
~/foto_manual/emisi2016/
├── 1000/   foto1.jpg foto2.jpg ...
├── 2000/
```
└── ... (100000/)
```

Output:
- `data/classification/{train,val,test}/{1000..100000}/`
- `manifest.json` - jejak tiap gambar (grup, sumber, split). Ini yang dipakai
  preflight check untuk mendeteksi kebocoran.
- `dataset_summary.json`

Flag penting:

| Flag | Fungsi |
|---|---|
| `--dedup-threshold` | Jarak Hamming pHash. 4 = agresif (buang near-dupe), 8 = sangat agresif, -1 = matikan |
| `--min-blur` | Ambang variance of Laplacian. Crop lebih blur dari ini dibuang |
| `--padding` | Padding dasar bbox; sistem menaikkannya otomatis untuk crop kecil |
| `--verify-leakage` | Cek ulang pasca-split (lambat tapi worth it sekali jalan) |

### Step 0b - Preflight check

```bash
python scripts/00b_preflight_check.py --data data/classification --deep
```

Wajib exit 0 sebelum lanjut. Yang dicek:

1. Struktur folder & jumlah per kelas
2. Rasio ketidakseimbangan kelas
3. **Kebocoran grup antar split** (dari `manifest.json`)
4. **Kebocoran perceptual** (gambar train yang nyaris identik dengan val/test)
5. Statistik blur - kalau train terlalu bersih, itu red flag
6. Statistik brightness - kalau variasinya sempit, artinya semua foto satu kondisi cahaya
7. Aspect ratio per kelas dibandingkan ukuran fisik resmi BI

Kalau muncul warning soal variasi pencahayaan sempit, itu artinya augmentasi saja
tidak cukup. Foto uang asli di warung remang dan di bawah lampu kuning tetap
kontribusi terbesar.

### Step 1 - Training

```bash
python scripts/01_train.py \
    --data data/classification \
    --output models \
    --aug-strength medium \
    --head-epochs 12 \
    --finetune-epochs 45 \
    --batch-size 32
```

Kalau selisih akurasi test bersih vs hard-test masih > 15 poin:

```bash
python scripts/01_train.py --aug-strength heavy --finetune-epochs 60
```

Output:
- `models/rupiah_final.keras` - output **logits** (untuk kalibrasi)
- `models/rupiah_infer.keras` - output **softmax** (untuk ekspor TFLite)
- `models/rupiah_mobilenetv2/` - SavedModel
- `models/test_logits.npz` - dipakai step 3
- `models/results.json`, `training_history.png`, `reliability_test.png`

### Step 2 - Kalibrasi & ambang tolak

```bash
python scripts/03_calibrate_threshold.py --models models --target-precision 0.995
```

Menghasilkan `models/calibration.json` berisi `temperature`,
`confidence_threshold`, `entropy_threshold`. Salin ke
`assets/models/calibration.json` di app Flutter.

### Step 3 - Export TFLite

```bash
python scripts/02_export_tflite.py \
    --model models/rupiah_infer.keras \
    --data data/classification \
    --output models/tflite \
    --rep-samples 400
```

---

## Deploy ke Flutter

```
models/tflite/rupiah_classifier_int8.tflite -> assets/models/uang_rupiah.tflite
models/tflite/labels.txt                    -> assets/models/rupiah_labels.txt
models/calibration.json                     -> assets/models/calibration.json
```

**Kontrak preprocessing yang WAJIB sama persis di sisi Dart:**

1. Ambil crop uang (atau frame penuh)
2. **Letterbox** ke 224x224: resize jaga aspect ratio, pad sisanya dengan 0
   (hitam), gambar di tengah. **Jangan** resize langsung ke persegi.
3. Channel order **RGB** (bukan BGR)
4. Normalisasi `x / 127.5 - 1.0` -> rentang `[-1, 1]`

Varian `int8.tflite` yang direkomendasikan punya **input/output float32**,
jadi kamu tidak perlu urus `scale`/`zero_point` di Dart. Kalau butuh full-int8
(misal untuk delegate NNAPI), pakai `rupiah_classifier_int8_full.tflite` dan
baca parameter quantization-nya dari output `02_export_tflite.py`.

Logika reject di sisi Dart:

```dart
final probs = runModel(letterboxed);          // 7 nilai, jumlah = 1
final conf = probs.reduce(math.max);
final entropy = -probs.fold<double>(
    0.0, (s, p) => s + p * math.log(p + 1e-9));

if (conf < calib.confidenceThreshold || entropy > calib.entropyThreshold) {
  ttsQueue.enqueue(
    "Uang belum terbaca jelas. Coba dekatkan dan ratakan uangnya.",
    priority: Priority.warning,
  );
} else {
  final idx = probs.indexOf(conf);
  ttsQueue.enqueue(calib.spoken[labels[idx]]!, priority: Priority.info);
}
```

Kenapa reject penting: salah baca nominal punya konsekuensi finansial nyata.
"Maaf, coba lagi" jauh lebih baik daripada "seratus ribu" padahal sepuluh ribu.

---

## Apa saja yang berubah, per file

### `00_merge_and_crop.py`

| Aspek | Lama | Baru |
|---|---|---|
| Split | `random.shuffle` per crop | Per grup (foto sumber), distratifikasi per kelas |
| Duplikat | Tidak ada penanganan | pHash 64-bit + bucketing, near-dupe dibuang |
| Padding | Fixed 5% | Adaptif: crop kecil dapat padding relatif lebih besar |
| Aspect ratio | Tidak diperhatikan | Crop disimpan apa adanya, letterbox saat training |
| Quality gate | Tidak ada | Filter blur (variance of Laplacian) dan ukuran minimum |
| Foto manual | Tidak didukung | `--extra-dirs` untuk foto emisi 2016/2022 |
| Audit trail | Tidak ada | `manifest.json` + `dataset_summary.json` |

### `00b_preflight_check.py`

Dari sekadar "folder ada dan tidak kosong" jadi 7 pemeriksaan, termasuk dua jenis
deteksi kebocoran dan statistik distribusi citra.

### `01_train.py`

| Aspek | Lama | Baru |
|---|---|---|
| Resize | `tf.image.resize` (squash ke persegi) | `resize_with_pad` (letterbox) |
| Flip | `random_flip_left_right` + `up_down` | Tanpa horizontal flip; rotasi bebas 0-360 |
| Hue | Sengaja dihindari total | Terbatas +/- 8 derajat |
| Deformasi | Tidak ada | Elastic, grid distortion, perspective, lipatan, kusut |
| Degradasi | Tidak ada | Coretan, stempel, noda, sobek, motion blur, JPEG artifact, downscale |
| Loss | `sparse_categorical_crossentropy` | `CategoricalCrossentropy` + label smoothing 0.05 |
| Regularisasi | Dropout saja | + MixUp/CutMix, weight decay, EMA |
| Imbalance | `class_weight` | Balanced sampling (kompatibel one-hot) |
| LR | Fixed + ReduceLROnPlateau | Warmup linear + cosine decay |
| Fine-tune | 80 layer terakhir, LR 1e-5 | Semua layer, LR 4e-5, BatchNorm dibekukan |
| Output | Softmax | Logits (+ model softmax terpisah untuk ekspor) |
| Evaluasi | Akurasi agregat | Per-kelas P/R/F1, confusion matrix, ECE, hard-test sintetis |

### `02_export_tflite.py`

| Aspek | Lama | Baru |
|---|---|---|
| Rep dataset | `tf.image.resize` (beda dari training) | Letterbox, identik dengan training |
| Sampling rep | Acak seluruh train | Seimbang per kelas |
| Varian INT8 | Satu (I/O tidak dispesifikasi) | Dua: float-IO (deploy) dan full-int8 (NNAPI) |
| Verifikasi | Akurasi agregat pada 50 gambar val | Per-kelas FP32 vs INT8, deteksi kelas yang jatuh |
| Kontrak I/O | Tidak dicetak | Scale & zero_point dicetak untuk disalin ke Dart |

### `03_calibrate_threshold.py` (baru)

Temperature scaling, sweep ambang confidence, ambang entropi lapis kedua,
kurva risk-coverage, dan analisis kekeliruan yang masih lolos ambang.

---

## Urutan prioritas kalau waktumu terbatas

1. **Regenerate dataset dengan script 00 baru.** Tanpa ini semua angka lain
   tidak bisa dipercaya, dan kamu tidak akan tahu apakah perbaikanmu berhasil.
2. **Jalankan 03_calibrate_threshold.py dan pasang reject option di app.**
   Ini memberi perlindungan langsung ke pengguna bahkan sebelum modelnya membaik.
3. **Training dengan augmentasi `medium`.** Ini yang menaikkan robustness.
4. **Kumpulkan foto uang lecek asli.** Target awal yang realistis: 50-100 foto
   per nominal dalam kondisi beragam (lecek, terlipat, dicoret, warung remang,
   lampu kuning, dipegang tangan). Augmentasi sintetis membantu, tapi tidak bisa
   menggantikan distribusi asli.

---

## Catatan performa

`tf.numpy_function` yang membungkus Albumentations menjalankan kode Python, jadi
terikat GIL. Sebagian besar operasi OpenCV melepas GIL sehingga throughput tetap
memadai dengan `num_parallel_calls=AUTOTUNE`, tapi kalau kamu lihat GPU idle
dan CPU 100%:

- turunkan `--aug-strength` ke `light`
- naikkan `--batch-size`
- atau cache decode gambar (butuh disk lebih banyak)

Di CPU murni, training ini akan lambat sekali. Pakai GPU (Colab/Kaggle cukup).

## Catatan yang perlu diverifikasi sendiri

- Angka perkiraan penurunan akurasi setelah perbaikan split (92-96%) itu
  dugaan berdasarkan pola umum dataset yang bocor, bukan hasil pengukuran di
  dataset kamu. Jalankan dan lihat sendiri.
- Mapping index kelas Roboflow di `src/common.py` (`ROBOFLOW_IDX_TO_CLASS`)
  diambil dari kode lama kamu. Cek ulang `data.yaml` tiap dataset sebelum
  jalan produksi; kalau ada dataset yang urutannya beda, hasilnya kacau tanpa
  error apa pun.
- Nilai default ambang (`--min-blur 25`, `--dedup-threshold 4`) adalah titik
  awal yang masuk akal, bukan angka optimal universal. Sesuaikan dengan
  melihat output preflight check.
