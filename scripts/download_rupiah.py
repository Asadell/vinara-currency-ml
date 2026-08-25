#!/usr/bin/env python3
import os
from roboflow import Roboflow

BASE_DIR = os.path.expanduser("~/datasets/rupiah-detection")
os.makedirs(BASE_DIR, exist_ok=True)

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

# (workspace, project, version, format, folder_name)
rupiah_datasets = [
    ("skripsi-3kth2", "deteksi-mata-uang-rupiah-nerog", 2, "yolov8", "rf-rupiah-skripsi"),      # 1,076 img, 7 kelas
    ("rupiah-detector", "rupiah-detector-qzmb7", 2, "yolov8", "rf-rupiah-detector"),             # 1,142 img, 7 kelas, mAP 98.8%
    ("workspace1-u35mt", "money-detection-valid", 4, "yolov8", "rf-money-detection-valid"),      # 3,791 img, 8 kelas (paling besar)
]

print("=" * 50)
print("DOWNLOADING ROBOFLOW RUPIAH/MONEY DATASETS")
print("=" * 50)
for workspace, project, version, fmt, folder_name in rupiah_datasets:
    target = os.path.join(BASE_DIR, folder_name)
    print(f"\n>>> Downloading {workspace}/{project} v{version} ({fmt}) ...")
    try:
        proj = rf.workspace(workspace).project(project)
        proj.version(version).download(fmt, location=target)
    except Exception as e:
        print(f"!!! FAILED: {workspace}/{project} - {e}")
        print("    (cek nomor versi terbaru di tab Versions halaman project itu)")

print("\n" + "=" * 50)
print("DONE. Cek ~/datasets/rupiah-detection/ untuk hasilnya.")
print("=" * 50)