#!/usr/bin/env bash
# ==============================================================================
# Rupiah Vision - Training Pipeline (Revised)
# MobileNetV2 Classifier -> TFLite
#
#   bash scripts/run_vast.sh
#   EPOCHS=60 bash scripts/run_vast.sh
#   BG_DIR=~/foto_latar bash scripts/run_vast.sh
#
# CATATAN: versi sebelumnya script ini TIDAK BISA JALAN sama sekali. Semua
# nama flag-nya salah (--datasets ke 00b, --data-dir/--epochs ke 01,
# --calib-dir ke 02, --model/--test-dir ke 03) dan path model hasilnya
# menunjuk runs/rupiah/*/best_model.keras yang tidak pernah dibuat script
# mana pun. Dengan `set -euo pipefail` dia abort di langkah kedua.
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── Python: deteksi venv atau fallback ke python3 ─────────────────────────────
if [ -f "/venv/main/bin/python" ]; then
    PYTHON_BIN="/venv/main/bin/python"
elif [ -f "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# ── Parameter ─────────────────────────────────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/rupiah-detection}"
DATA_DIR="${DATA_DIR:-data/classification}"
OUT_DIR="${OUT_DIR:-models}"
HEAD_EPOCHS="${HEAD_EPOCHS:-12}"
EPOCHS="${EPOCHS:-45}"
BATCH_SIZE="${BATCH_SIZE:-32}"
AUG_STRENGTH="${AUG_STRENGTH:-medium}"
FRAME_PROB="${FRAME_PROB:-0.75}"
FRAME_SCALE_MIN="${FRAME_SCALE_MIN:-0.28}"
# Folder foto LATAR (tangan, meja, keyboard, lantai, kain) TANPA uang.
# Kosongkan kalau belum punya, nanti dipakai latar prosedural yang jauh lemah.
BG_DIR="${BG_DIR:-}"
# Folder foto manual per nominal: <dir>/1000, <dir>/2000, ... <dir>/100000
EXTRA_DIRS="${EXTRA_DIRS:-}"

echo "======================================================================"
echo "  Rupiah Vision Training - Revised Pipeline"
echo "  Repo     : $REPO_DIR"
echo "  Python   : $PYTHON_BIN"
echo "  Data     : $DATA_DIR"
echo "  Augment  : $AUG_STRENGTH   frame-prob=$FRAME_PROB"
echo "  Latar    : ${BG_DIR:-(prosedural)}"
echo "======================================================================"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo -e "\n[PRE-FLIGHT] Cek & install dependencies..."
"$PYTHON_BIN" -m pip install -r requirements.txt -q
"$PYTHON_BIN" - <<'PY'
import albumentations as A, tensorflow as tf
print(f"  albumentations {A.__version__} | tensorflow {tf.__version__}")
# Gagal cepat kalau API-nya tidak cocok. Lebih baik meledak di sini daripada
# augmentasi mati diam-diam selama 45 epoch.
A.Compose([
    A.CoarseDropout(num_holes_range=(1, 2), fill=0, fill_mask=0, p=1.0),
    A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
    A.ToGray(p=1.0),
    A.CLAHE(clip_limit=(1.0, 3.0), p=1.0),
])
print("  API albumentations cocok")
PY

# ── Step 1: download dataset ──────────────────────────────────────────────────
if [ ! -d "$DATASET_DIR/rf-rupiah-detector" ]; then
    echo -e "\n[1/5] Download dataset ke $DATASET_DIR ..."
    "$PYTHON_BIN" scripts/download_rupiah.py
else
    echo -e "\n[1/5] Dataset sudah ada di $DATASET_DIR"
fi

# ── Step 2: merge, crop, dedup, group-aware split ─────────────────────────────
echo -e "\n[2/5] Merge & crop -> $DATA_DIR ..."
MERGE_ARGS=(
    --output "$DATA_DIR"
    --val-split 0.15
    --test-split 0.10
    --dedup-threshold 4
    --min-blur 25
    --verify-leakage
    --clean
)
if [ -n "$EXTRA_DIRS" ]; then
    # shellcheck disable=SC2206
    MERGE_ARGS+=(--extra-dirs ${EXTRA_DIRS})
fi
"$PYTHON_BIN" scripts/00_merge_and_crop.py "${MERGE_ARGS[@]}"

# ── Step 3: preflight (wajib exit 0) ──────────────────────────────────────────
echo -e "\n[3/5] Preflight check..."
"$PYTHON_BIN" scripts/00b_preflight_check.py --data "$DATA_DIR" --deep

# ── Step 4: training ──────────────────────────────────────────────────────────
echo -e "\n[4/5] Training MobileNetV2 (head=$HEAD_EPOCHS, finetune=$EPOCHS)..."
TRAIN_ARGS=(
    --data "$DATA_DIR"
    --output "$OUT_DIR"
    --img-size 224
    --batch-size "$BATCH_SIZE"
    --head-epochs "$HEAD_EPOCHS"
    --finetune-epochs "$EPOCHS"
    --aug-strength "$AUG_STRENGTH"
    --frame-prob "$FRAME_PROB"
    --frame-scale-min "$FRAME_SCALE_MIN"
)
if [ -n "$BG_DIR" ]; then
    TRAIN_ARGS+=(--bg-dir "$BG_DIR")
fi
"$PYTHON_BIN" scripts/01_train.py "${TRAIN_ARGS[@]}"

# ── Step 5: kalibrasi + export ────────────────────────────────────────────────
echo -e "\n[5/5] Kalibrasi ambang + export TFLite..."
"$PYTHON_BIN" scripts/03_calibrate_threshold.py \
    --models "$OUT_DIR" \
    --target-precision 0.995

"$PYTHON_BIN" scripts/02_export_tflite.py \
    --model "$OUT_DIR/rupiah_infer.keras" \
    --data "$DATA_DIR" \
    --output "$OUT_DIR/tflite" \
    --rep-samples 400

# ── Ringkasan ─────────────────────────────────────────────────────────────────
echo -e "\n======================================================================"
echo "  SELESAI. Output:"
echo "    $OUT_DIR/tflite/            (fp32 / fp16 / int8)"
echo "    $OUT_DIR/calibration.json   (ambang tolak)"
echo "    $OUT_DIR/results.json       (metrik, termasuk COLOR-STRESS)"
echo ""
echo "  YANG WAJIB DILIHAT DULUAN di results.json:"
echo "    test_grayscale.accuracy"
echo "      < 23%  -> model masih classifier warna murni, 20rb vs 50rb"
echo "                akan tetap ketuker. Naikkan decolor / pakai heavy."
echo "      > 55%  -> model sudah punya isyarat non-warna. Lanjut deploy."
echo "    augmentation_health.ratio"
echo "      > 2%   -> augmentasi banyak gagal, angka lain tidak bisa dipercaya."
echo ""
echo "  Deploy ke Flutter:"
echo "    $OUT_DIR/tflite/rupiah_classifier_fp16.tflite"
echo "    $OUT_DIR/calibration.json"
echo "    -> project/guidio_app/assets/models/"
echo "======================================================================"
