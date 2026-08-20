#!/usr/bin/env bash
# ==============================================================================
# Rupiah Vision — Training Pipeline (Revised)
# MobileNetV2 Classifier -> TFLite INT8
#
# Usage di VPS setelah setup credentials:
#   bash scripts/run_vast.sh
#   bash scripts/run_vast.sh 60   # custom epochs
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── Python: deteksi venv atau fallback ke python3 ─────────────────────────────
PYTHON_BIN=""
if [ -f "/venv/main/bin/python" ]; then
    PYTHON_BIN="/venv/main/bin/python"
elif [ -f "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "  Rupiah Vision Training — Revised Pipeline"
echo "  Repo     : $REPO_DIR"
echo "  Python   : $PYTHON_BIN"
echo "======================================================================"

# ── Cek & install dependencies ────────────────────────────────────────────────
echo -e "\n[PRE-FLIGHT] Checking dependencies..."
"$PYTHON_BIN" -c "import tensorflow, albumentations, imagehash, cv2" 2>/dev/null || {
    echo "  Installing from requirements.txt ..."
    "$PYTHON_BIN" -m pip install -r requirements.txt -q
}
echo "  ✓ Dependencies OK"

# ── Parameter ─────────────────────────────────────────────────────────────────
EPOCHS="${1:-50}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/rupiah-detection}"

# ── Step 1: Download dataset Roboflow ─────────────────────────────────────────
if [ ! -d "$DATASET_DIR/rf-rupiah-detector" ]; then
    echo -e "\n[1/5] Downloading rupiah datasets ke $DATASET_DIR ..."
    "$PYTHON_BIN" ../datasets/download_rupiah.py
else
    echo -e "\n[1/5] Dataset sudah ada di $DATASET_DIR — skip download"
fi

# ── Step 2: Preflight check ───────────────────────────────────────────────────
echo -e "\n[2/5] Running preflight check..."
"$PYTHON_BIN" scripts/00b_preflight_check.py \
    --datasets "$DATASET_DIR/rf-rupiah-detector" \
               "$DATASET_DIR/rf-money-detection-valid" \
               "$DATASET_DIR/rf-rupiah-skripsi"

# ── Step 3: Merge, crop, dedup, group-aware split ─────────────────────────────
echo -e "\n[3/5] Merge & crop dataset -> data/classification ..."
"$PYTHON_BIN" scripts/00_merge_and_crop.py \
    --datasets "$DATASET_DIR/rf-rupiah-detector" \
               "$DATASET_DIR/rf-money-detection-valid" \
               "$DATASET_DIR/rf-rupiah-skripsi" \
    --output data/classification \
    --val-split 0.15 \
    --test-split 0.10 \
    --dedup-threshold 4 \
    --min-blur 25 \
    --verify-leakage \
    --clean

# ── Step 4: Training MobileNetV2 ──────────────────────────────────────────────
echo -e "\n[4/5] Training MobileNetV2 ($EPOCHS epochs)..."
"$PYTHON_BIN" scripts/01_train.py \
    --data-dir data/classification \
    --epochs "$EPOCHS" \
    --batch-size 32 \
    --img-size 224

# ── Step 5: Export TFLite INT8 + kalibrasi threshold ─────────────────────────
echo -e "\n[5/5] Exporting TFLite INT8 + calibrating thresholds..."

BEST_MODEL=$(ls -t runs/rupiah/*/best_model.keras 2>/dev/null | head -1 || echo "")
if [ -f "$BEST_MODEL" ]; then
    "$PYTHON_BIN" scripts/02_export_tflite.py \
        --model "$BEST_MODEL" \
        --calib-dir data/classification/val \
        --output runs/rupiah_classifier_int8.tflite

    "$PYTHON_BIN" scripts/03_calibrate_threshold.py \
        --model runs/rupiah_classifier_int8.tflite \
        --test-dir data/classification/test \
        --output runs/rupiah_thresholds.json
else
    echo "  ⚠ Trained model tidak ditemukan — skip export"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "\n======================================================================"
echo "  [✓] SELESAI! Output ada di:"
echo ""
echo "  Mobile (Flutter):"
echo "    runs/rupiah_classifier_int8.tflite"
echo "    runs/rupiah_thresholds.json"
echo ""
echo "  Langkah selanjutnya:"
echo "    1. Salin rupiah_classifier_int8.tflite ke:"
echo "       project/guidio_app/assets/models/"
echo "    2. Update MoneyTFLiteService jika ada perubahan kelas/threshold"
echo "======================================================================"
