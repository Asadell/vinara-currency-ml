#!/usr/bin/env python3
import os
from roboflow import Roboflow

BASE_DIR = os.path.expanduser("~/datasets/rupiah-detection")
os.makedirs(BASE_DIR, exist_ok=True)

rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", ""))

# (workspace, project, default_version, format, folder_name)
rupiah_datasets = [
    # 3 Dataset awal
    ("skripsi-3kth2", "deteksi-mata-uang-rupiah-nerog", 2, "yolov8", "rf-rupiah-skripsi"),
    ("rupiah-detector", "rupiah-detector-qzmb7", 2, "yolov8", "rf-rupiah-detector"),
    ("workspace1-u35mt", "money-detection-valid", 4, "yolov8", "rf-money-detection-valid"),
    # 13 Dataset tambahan
    ("moneysaver-yolo", "deteksi-uang-s0pfe", 1, "yolov8", "rf-moneysaver-yolo"),
    ("skripsi-swuyl", "detetksi-keaslian-uang", 1, "yolov8", "rf-skripsi-swuyl"),
    ("yoloai-3iuvz", "uang_deteksi", 1, "yolov8", "rf-yoloai-3iuvz"),
    ("amndan", "uangbaru2022", 1, "yolov8", "rf-amndan"),
    ("jemy07s-workspace", "tubes-psi-ydfzm", 1, "yolov8", "rf-jemy07s"),
    ("adelias-workspace", "modeluangv2-t1lk9", 1, "yolov8", "rf-adelias"),
    ("zannho", "rupiah-detection-o3agm", 1, "yolov8", "rf-zannho"),
    ("muhammad-aidil-wlsfe", "cnnyolo-hhphe", 1, "yolov8", "rf-muhammad-aidil"),
    ("cahyadin", "money_detection-pnnd7", 1, "yolov8", "rf-cahyadin"),
    ("4ia17ottos-workspace", "moneydetection-uetq8", 1, "yolov8", "rf-4ia17ottos-uetq8"),
    ("tes-nms6d", "uang-kertas-2022-dan-logam-2016", 1, "yolov8", "rf-tes-nms6d"),
    ("4ia17ottos-workspace", "uang_baru", 1, "yolov8", "rf-4ia17ottos-uang-baru"),
    ("project-binus", "rupiah-detection-d1vbz", 1, "yolov8", "rf-project-binus"),
]

print("=" * 60)
print("DOWNLOADING 16 ROBOFLOW RUPIAH/MONEY DATASETS")
print("=" * 60)
for workspace, project, version, fmt, folder_name in rupiah_datasets:
    target = os.path.join(BASE_DIR, folder_name)
    print(f"\n>>> Downloading {workspace}/{project} (v{version}) ...")
    try:
        proj = rf.workspace(workspace).project(project)
        try:
            ver = proj.version(version)
        except Exception:
            ver = proj.version(1)
        ver.download(fmt, location=target)
        print(f"✅ Selesai: {folder_name}")
    except Exception as e:
        print(f"!!! FAILED: {workspace}/{project} - {e}")

print("\n" + "=" * 60)
print("SELESAI. Semua dataset tersimpan di ~/datasets/rupiah-detection/")
print("=" * 60)