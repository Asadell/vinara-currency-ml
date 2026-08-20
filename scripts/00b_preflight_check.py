#!/usr/bin/env python3
"""
00b_preflight_check.py  (REVISI)
================================
Verifikasi dataset SEBELUM training. Harus exit code 0 sebelum 01_train.py.

Versi lama cuma mengecek "folder ada dan tidak kosong". Itu tidak cukup:
dataset bisa lolos check tapi tetap menghasilkan model yang kelihatan
99.5% padahal bocor. Versi ini menambah pemeriksaan yang benar-benar
berkorelasi dengan kualitas model:

  1. Struktur folder & jumlah gambar per kelas per split
  2. Rasio ketidakseimbangan kelas (imbalance ratio)
  3. Kebocoran GRUP antar split (dari manifest.json)
  4. Kebocoran perceptual (gambar train mirip gambar val/test)
  5. Statistik blur per split (train terlalu bersih = red flag)
  6. Statistik aspect ratio per kelas (deteksi crop yang salah)
  7. Statistik brightness (deteksi dataset yang cuma satu kondisi cahaya)
  8. Gambar rusak / tidak terbaca

Exit code:
  0 = lolos (boleh ada peringatan)
  1 = ada error fatal, training akan menghasilkan model yang menyesatkan

Usage:
    python scripts/00b_preflight_check.py --data data/classification
    python scripts/00b_preflight_check.py --data data/classification --deep
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    CLASS_ORDER,
    NOMINAL_ASPECT,
    blur_score,
    hamming,
    list_images,
    phash_64,
)

SPLITS = ["train", "val", "test"]

# Ambang minimum yang masuk akal untuk kasus GUIDIO
MIN_TRAIN_PER_CLASS = 150
MIN_EVAL_PER_CLASS = 25
MAX_IMBALANCE_RATIO = 8.0


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def note(self, msg: str):
        self.notes.append(msg)


# ─── Cek 1-2: struktur & keseimbangan ──────────────────────────────────────────

def check_structure(data_dir: Path, rep: Report) -> dict:
    print("[1] Struktur folder & jumlah gambar")
    counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}

    if not data_dir.exists():
        rep.error(f"Folder data tidak ada: {data_dir.resolve()}")
        return counts

    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            rep.error(f"Split folder hilang: {split_dir}")
            continue
        total = 0
        for cls in CLASS_ORDER:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                rep.error(f"Folder kelas hilang: {cls_dir}")
                counts[split][cls] = 0
                continue
            n = len(list_images(cls_dir))
            counts[split][cls] = n
            total += n
            if n == 0:
                rep.error(f"Kosong: {split}/{cls}")
        print(f"    {split:5s}: {total:6d} gambar")

    print(f"\n    {'Kelas':>10} {'train':>8} {'val':>8} {'test':>8}")
    print(f"    {'-' * 38}")
    for cls in CLASS_ORDER:
        print(f"    {cls:>10} {counts['train'].get(cls, 0):>8} "
              f"{counts['val'].get(cls, 0):>8} {counts['test'].get(cls, 0):>8}")
    return counts


def check_balance(counts: dict, rep: Report) -> None:
    print("\n[2] Keseimbangan kelas")
    train = counts.get("train", {})
    vals = [v for v in train.values() if v > 0]
    if not vals:
        rep.error("Train set kosong total.")
        return

    lo, hi = min(vals), max(vals)
    ratio = hi / lo
    print(f"    min={lo}  max={hi}  imbalance ratio={ratio:.2f}x")

    for cls in CLASS_ORDER:
        if train.get(cls, 0) < MIN_TRAIN_PER_CLASS:
            rep.warn(f"train/{cls} cuma {train.get(cls, 0)} gambar "
                     f"(disarankan minimal {MIN_TRAIN_PER_CLASS})")
        for split in ("val", "test"):
            if counts.get(split, {}).get(cls, 0) < MIN_EVAL_PER_CLASS:
                rep.warn(f"{split}/{cls} cuma {counts[split].get(cls, 0)} gambar "
                         f"(disarankan minimal {MIN_EVAL_PER_CLASS}); "
                         f"metrik per-kelas jadi tidak stabil")

    if ratio > MAX_IMBALANCE_RATIO:
        rep.warn(f"Imbalance {ratio:.1f}x cukup parah. "
                 f"01_train.py sudah pakai balanced sampling, tapi tetap "
                 f"lebih baik tambah data untuk kelas minoritas.")


# ─── Cek 3: kebocoran grup ─────────────────────────────────────────────────────

def check_group_leakage(data_dir: Path, rep: Report) -> None:
    print("\n[3] Kebocoran grup antar split")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        rep.warn("manifest.json tidak ada. Kemungkinan dataset dibuat "
                 "dengan versi lama 00_merge_and_crop.py yang split-nya "
                 "acak per gambar (rawan kebocoran). Sangat disarankan "
                 "regenerate dataset dengan script 00 versi baru.")
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        rep.warn(f"manifest.json tidak terbaca: {e}")
        return

    group_splits: dict[str, set] = defaultdict(set)
    for m in manifest:
        group_splits[m["group"]].add(m["split"])

    bocor = {g: s for g, s in group_splits.items() if len(s) > 1}
    if bocor:
        rep.error(f"{len(bocor)} grup muncul di lebih dari satu split. "
                  f"Ini kebocoran langsung. Contoh: "
                  f"{list(bocor.items())[:3]}")
    else:
        print(f"    Bersih: {len(group_splits)} grup, tidak ada yang lintas split.")


# ─── Cek 4: kebocoran perceptual ───────────────────────────────────────────────

def check_perceptual_leakage(data_dir: Path, rep: Report,
                             sample_per_class: int, threshold: int) -> None:
    print("\n[4] Kebocoran perceptual (train vs val/test)")
    rng = random.Random(1234)

    train_hashes: list[int] = []
    for cls in CLASS_ORDER:
        imgs = list_images(data_dir / "train" / cls)
        rng.shuffle(imgs)
        for p in imgs[:sample_per_class]:
            img = cv2.imread(str(p))
            if img is not None:
                train_hashes.append(phash_64(img))

    if not train_hashes:
        rep.warn("Tidak ada gambar train yang bisa dibaca untuk cek ini.")
        return

    n_hits = 0
    n_checked = 0
    examples: list[str] = []
    for split in ("val", "test"):
        for cls in CLASS_ORDER:
            imgs = list_images(data_dir / split / cls)
            rng.shuffle(imgs)
            for p in imgs[:sample_per_class]:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h = phash_64(img)
                n_checked += 1
                for th in train_hashes:
                    if hamming(h, th) <= threshold:
                        n_hits += 1
                        if len(examples) < 5:
                            examples.append(f"{split}/{cls}/{p.name}")
                        break

    if n_checked == 0:
        rep.warn("Tidak ada gambar val/test yang bisa dicek.")
        return

    pct = 100.0 * n_hits / n_checked
    print(f"    Dicek {n_checked} gambar eval terhadap {len(train_hashes)} "
          f"gambar train (sampel).")
    print(f"    Nyaris identik dengan train: {n_hits} ({pct:.1f}%)")

    if pct > 10:
        rep.error(f"{pct:.1f}% gambar eval nyaris identik dengan train. "
                  f"Test accuracy kamu TIDAK bisa dipercaya. "
                  f"Contoh: {examples}")
    elif pct > 3:
        rep.warn(f"{pct:.1f}% gambar eval mirip train. Naikkan "
                 f"--dedup-threshold saat regenerate dataset. Contoh: {examples}")
    else:
        print("    Aman.")


# ─── Cek 5-7: statistik citra ──────────────────────────────────────────────────

def check_image_stats(data_dir: Path, rep: Report, sample_per_class: int) -> None:
    print("\n[5-7] Statistik citra (blur, brightness, aspect ratio)")
    rng = random.Random(99)
    n_broken = 0

    for split in SPLITS:
        blurs: list[float] = []
        brights: list[float] = []
        for cls in CLASS_ORDER:
            imgs = list_images(data_dir / split / cls)
            rng.shuffle(imgs)
            for p in imgs[:sample_per_class]:
                img = cv2.imread(str(p))
                if img is None:
                    n_broken += 1
                    continue
                blurs.append(blur_score(img))
                brights.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))

        if not blurs:
            continue
        b = np.array(blurs)
        v = np.array(brights)
        print(f"    {split:5s}  blur: p10={np.percentile(b, 10):7.1f} "
              f"med={np.median(b):7.1f} p90={np.percentile(b, 90):7.1f}   "
              f"brightness: med={np.median(v):5.1f} std={v.std():5.1f}")

        if split == "train" and np.percentile(b, 10) > 150:
            rep.warn("Train set nyaris tidak punya gambar blur sama sekali. "
                     "Model bakal kaget lihat foto HP goyang. Augmentasi blur "
                     "di 01_train.py membantu, tapi foto asli yang agak blur "
                     "lebih baik lagi.")
        if split == "train" and v.std() < 22:
            rep.warn("Variasi pencahayaan train sangat sempit "
                     f"(std={v.std():.1f}). Ini indikasi semua foto diambil "
                     "pada kondisi cahaya serupa. Tambah foto kondisi remang, "
                     "lampu kuning, dan backlight.")

    if n_broken:
        rep.error(f"{n_broken} gambar tidak bisa dibaca (file rusak).")

    # Aspect ratio per kelas
    print("\n    Aspect ratio (panjang/lebar) per kelas di train:")
    print(f"    {'Kelas':>10} {'median':>8} {'ideal':>8} {'selisih':>9}")
    print(f"    {'-' * 40}")
    for cls in CLASS_ORDER:
        imgs = list_images(data_dir / "train" / cls)
        rng.shuffle(imgs)
        ratios = []
        for p in imgs[:sample_per_class]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            ratios.append(max(h, w) / max(1, min(h, w)))
        if not ratios:
            continue
        med = float(np.median(ratios))
        ideal = NOMINAL_ASPECT[cls]
        diff = abs(med - ideal)
        flag = "  <-- cek crop" if diff > 0.45 else ""
        print(f"    {cls:>10} {med:>8.2f} {ideal:>8.2f} {diff:>9.2f}{flag}")
        if diff > 0.45:
            rep.warn(f"Aspect ratio median kelas {cls} ({med:.2f}) jauh dari "
                     f"ukuran fisik seharusnya ({ideal:.2f}). Kemungkinan "
                     f"bbox kepotong, uang terlipat, atau padding kebesaran.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pre-flight check dataset rupiah-vision")
    ap.add_argument("--data", default="data/classification")
    ap.add_argument("--sample-per-class", type=int, default=60,
                    help="Jumlah gambar per kelas yang disampel untuk statistik")
    ap.add_argument("--leak-threshold", type=int, default=4,
                    help="Jarak Hamming pHash untuk dianggap bocor")
    ap.add_argument("--deep", action="store_true",
                    help="Sampel lebih banyak (lebih akurat, lebih lambat)")
    ap.add_argument("--strict", action="store_true",
                    help="Perlakukan peringatan sebagai error")
    args = ap.parse_args()

    if args.deep:
        args.sample_per_class = max(args.sample_per_class, 250)

    data_dir = Path(args.data)
    rep = Report()

    print("=" * 62)
    print("  PRE-FLIGHT CHECK - Rupiah Vision")
    print("=" * 62)
    print(f"  Data dir: {data_dir.resolve()}\n")

    counts = check_structure(data_dir, rep)
    if rep.errors:
        print_report(rep)
        sys.exit(1)

    check_balance(counts, rep)
    check_group_leakage(data_dir, rep)
    check_perceptual_leakage(data_dir, rep, args.sample_per_class,
                             args.leak_threshold)
    check_image_stats(data_dir, rep, args.sample_per_class)

    fatal = print_report(rep)
    if fatal or (args.strict and rep.warnings):
        sys.exit(1)

    print("\n  Lanjut:")
    print("    python scripts/01_train.py --data data/classification \\")
    print("        --output models --aug-strength medium")
    sys.exit(0)


def print_report(rep: Report) -> bool:
    print("\n" + "=" * 62)
    if rep.errors:
        print("  HASIL: GAGAL")
    elif rep.warnings:
        print("  HASIL: LOLOS DENGAN CATATAN")
    else:
        print("  HASIL: LOLOS BERSIH")
    print("=" * 62)

    for e in rep.errors:
        print(f"  [ERROR] {e}")
    for w in rep.warnings:
        print(f"  [WARN ] {w}")
    for n in rep.notes:
        print(f"  [INFO ] {n}")

    if rep.errors:
        print("\n  Perbaiki error di atas dulu. Training di atas dataset "
              "yang bocor cuma menghasilkan angka bagus yang bohong.")
    return bool(rep.errors)


if __name__ == "__main__":
    main()
