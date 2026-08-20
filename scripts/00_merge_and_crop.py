#!/usr/bin/env python3
"""
00_merge_and_crop.py  (REVISI)
==============================
Merge dataset deteksi (YOLO/OBB) + folder foto manual -> dataset klasifikasi.

PERUBAHAN BESAR DARI VERSI LAMA (baca ini, penting banget):

  1. GROUP-AWARE SPLIT (perbaikan bug paling kritis)
     Versi lama: semua crop dikumpulkan lalu `random.shuffle` dan dipotong
     jadi train/val/test. Masalahnya, satu foto sumber sering berisi
     BEBERAPA lembar uang, dan dataset Roboflow banyak yang berasal dari
     frame video (foto nyaris identik berturut-turut). Akibatnya crop dari
     foto yang sama (atau nyaris sama) bocor ke train DAN test sekaligus.
     Model tinggal "menghafal" dan test accuracy jadi 99.5% padahal
     generalisasinya lemah. Ini penjelasan paling masuk akal kenapa akurasi
     lab tinggi tapi di lapangan jeblok.

     Versi baru: split dilakukan per GRUP (satu foto sumber = satu grup).
     Semua crop dari satu foto sumber dijamin masuk ke split yang sama.

  2. DEDUPLIKASI PERCEPTUAL HASH
     Buang near-duplicate (frame video berurutan, foto burst) sebelum split.
     Pakai pHash 64-bit + jarak Hamming.

  3. ADAPTIVE PADDING
     Padding bbox tidak lagi fixed 5%. Crop kecil dikasih padding relatif
     lebih besar (biar konteks tepi uang ikut), crop besar lebih kecil.

  4. SIMPAN ASPECT RATIO ASLI
     Crop TIDAK di-resize paksa ke persegi. Resize/letterbox dilakukan saat
     training. Rasio panjang:lebar Rupiah naik monoton dari 1.86 (1rb)
     ke 2.32 (100rb), jadi itu sinyal yang sayang kalau dibuang.

  5. DUKUNGAN FOTO MANUAL
     `--extra-dirs` menerima folder yang sudah tersusun sebagai
     <folder>/<nominal>/*.jpg (misal foto manual emisi 2016 & 2022).
     Tiap file dianggap satu grup sendiri.

  6. QUALITY GATE
     Crop yang terlalu blur (variance of Laplacian rendah) atau terlalu
     gelap/terang ekstrem bisa dibuang otomatis.

Usage:
    python scripts/00_merge_and_crop.py \
        --datasets ~/datasets/rf-rupiah-detector \
                   ~/datasets/rf-money-detection-valid \
                   ~/datasets/rf-rupiah-skripsi \
        --extra-dirs ~/foto_manual/emisi2016 ~/foto_manual/emisi2022 \
        --output data/classification \
        --val-split 0.15 --test-split 0.10 \
        --dedup-threshold 4 \
        --min-blur 25
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    CLASS_ORDER,
    IMG_EXTENSIONS,
    ROBOFLOW_IDX_TO_CLASS,
    blur_score,
    hamming,
    phash_64,
)


# ─── Parsing label ─────────────────────────────────────────────────────────────

def parse_label_line(line: str):
    """
    Parse satu baris file label YOLO/OBB.
    Return (class_idx, x_min, y_min, x_max, y_max) ternormalisasi [0,1],
    atau None kalau baris tidak valid.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(parts[0])
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    n = len(vals)
    if n == 4:
        cx, cy, w, h = vals
        x_min, y_min = cx - w / 2, cy - h / 2
        x_max, y_max = cx + w / 2, cy + h / 2
    elif n >= 6 and n % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    else:
        return None

    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    if x_max - x_min < 1e-4 or y_max - y_min < 1e-4:
        return None
    return cls, x_min, y_min, x_max, y_max


# ─── Crop ──────────────────────────────────────────────────────────────────────

def adaptive_padding(box_w_px: float, box_h_px: float,
                     base: float, min_pad_px: int = 6) -> tuple[float, float]:
    """
    Padding relatif adaptif.

    Crop kecil (uang jauh dari kamera) butuh padding relatif lebih besar
    supaya tepi uang tidak kepotong dan konteksnya ikut. Crop besar cukup
    padding kecil. Selain itu ada lantai minimum dalam pixel supaya crop
    super kecil tetap dapat margin yang berarti.
    """
    short_side = max(1.0, min(box_w_px, box_h_px))
    # Skala: crop < 100px dapat padding sampai 2x base, crop > 400px ~ base
    scale = float(np.clip(300.0 / short_side, 1.0, 2.5))
    pad_w = max(base * scale, min_pad_px / max(box_w_px, 1.0))
    pad_h = max(base * scale, min_pad_px / max(box_h_px, 1.0))
    return float(pad_w), float(pad_h)


def crop_bbox(img: np.ndarray, x_min, y_min, x_max, y_max,
              base_padding: float) -> np.ndarray:
    """Crop bbox dari gambar dengan padding relatif adaptif."""
    H, W = img.shape[:2]
    box_w_px = (x_max - x_min) * W
    box_h_px = (y_max - y_min) * H

    pad_rel_w, pad_rel_h = adaptive_padding(box_w_px, box_h_px, base_padding)
    pw = (x_max - x_min) * pad_rel_w
    ph = (y_max - y_min) * pad_rel_h

    x1 = max(0, int(round((x_min - pw) * W)))
    y1 = max(0, int(round((y_min - ph) * H)))
    x2 = min(W, int(round((x_max + pw) * W)))
    y2 = min(H, int(round((y_max + ph) * H)))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2]


def cap_size(img: np.ndarray, max_side: int) -> np.ndarray:
    """Batasi sisi terpanjang supaya file di disk tidak kebesaran."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


# ─── Dedup on-the-fly ──────────────────────────────────────────────────────────

class PhashDeduper:
    """
    Dedup near-duplicate memakai pHash + bucketing prefix.
    Bucket dipakai 12-bit prefix, dengan pengecekan bucket tetangga
    (flip 1 bit prefix) supaya near-dupe yang beda tipis di prefix
    tetap ketangkap.
    """

    def __init__(self, threshold: int = 4, prefix_bits: int = 12):
        self.threshold = threshold
        self.prefix_bits = prefix_bits
        self.shift = 64 - prefix_bits
        self.buckets: dict[int, list[int]] = defaultdict(list)
        self.n_removed = 0

    def _keys(self, h: int):
        base = h >> self.shift
        yield base
        for b in range(self.prefix_bits):
            yield base ^ (1 << b)

    def is_duplicate(self, h: int) -> bool:
        if self.threshold < 0:
            return False
        for key in self._keys(h):
            for other in self.buckets.get(key, ()):
                if hamming(h, other) <= self.threshold:
                    self.n_removed += 1
                    return True
        return False

    def add(self, h: int) -> None:
        self.buckets[h >> self.shift].append(h)


# ─── Pengumpulan ───────────────────────────────────────────────────────────────

def collect_from_detection_dataset(
    dataset_dir: Path,
    staging_dir: Path,
    deduper: PhashDeduper,
    base_padding: float,
    min_crop_size: int,
    min_blur: float,
    max_side: int,
    records: list,
    stats: Counter,
) -> None:
    """Scan satu folder dataset Roboflow YOLO, crop tiap bbox, simpan ke staging."""
    ds_name = dataset_dir.name

    for split in ("train", "valid", "val", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = sorted(
            f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS
        )
        print(f"   [{ds_name}/{split}] {len(img_files)} gambar sumber")

        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                stats["no_label"] += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                stats["unreadable"] += 1
                continue

            try:
                lines = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                stats["unreadable"] += 1
                continue

            # Satu foto sumber = satu grup. Ini kunci anti-leakage.
            group_key = f"{ds_name}::{split}::{img_path.stem}"

            for line_i, line in enumerate(lines):
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                cls_idx, x_min, y_min, x_max, y_max = parsed
                if cls_idx not in ROBOFLOW_IDX_TO_CLASS:
                    stats["unknown_class"] += 1
                    continue

                class_name = ROBOFLOW_IDX_TO_CLASS[cls_idx]
                crop = crop_bbox(img, x_min, y_min, x_max, y_max, base_padding)

                if crop.size == 0 or min(crop.shape[:2]) < min_crop_size:
                    stats["too_small"] += 1
                    continue

                if min_blur > 0 and blur_score(crop) < min_blur:
                    stats["too_blurry"] += 1
                    continue

                h = phash_64(crop)
                if deduper.is_duplicate(h):
                    stats["duplicate"] += 1
                    continue
                deduper.add(h)

                crop = cap_size(crop, max_side)
                uid = f"{len(records):06d}"
                out_path = staging_dir / class_name / f"{class_name}_{uid}.jpg"
                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                records.append({
                    "path": str(out_path),
                    "class": class_name,
                    "group": group_key,
                    "source": str(img_path),
                    "origin": ds_name,
                    "h": crop.shape[0],
                    "w": crop.shape[1],
                })
                stats["kept"] += 1


def collect_from_class_folders(
    root_dir: Path,
    staging_dir: Path,
    deduper: PhashDeduper,
    min_crop_size: int,
    min_blur: float,
    max_side: int,
    records: list,
    stats: Counter,
) -> None:
    """
    Scan folder foto manual yang sudah tersusun <root>/<nominal>/*.jpg.
    Cocok untuk foto manual emisi 2016 & 2022 yang kamu ambil sendiri.
    Tiap file = satu grup sendiri.
    """
    origin = root_dir.name
    for class_name in CLASS_ORDER:
        cls_dir = root_dir / class_name
        if not cls_dir.exists():
            continue
        files = sorted(
            f for f in cls_dir.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
        print(f"   [{origin}/{class_name}] {len(files)} foto manual")

        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                stats["unreadable"] += 1
                continue
            if min(img.shape[:2]) < min_crop_size:
                stats["too_small"] += 1
                continue
            if min_blur > 0 and blur_score(img) < min_blur:
                stats["too_blurry"] += 1
                continue

            h = phash_64(img)
            if deduper.is_duplicate(h):
                stats["duplicate"] += 1
                continue
            deduper.add(h)

            img = cap_size(img, max_side)
            uid = f"{len(records):06d}"
            out_path = staging_dir / class_name / f"{class_name}_{uid}.jpg"
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            records.append({
                "path": str(out_path),
                "class": class_name,
                "group": f"{origin}::manual::{f.stem}",
                "source": str(f),
                "origin": origin,
                "h": img.shape[0],
                "w": img.shape[1],
            })
            stats["kept"] += 1


# ─── Split ─────────────────────────────────────────────────────────────────────

def group_stratified_split(
    records: list,
    val_split: float,
    test_split: float,
    seed: int,
) -> dict[str, str]:
    """
    Bagi GRUP (bukan gambar) ke train/val/test, distratifikasi berdasarkan
    kelas dominan tiap grup supaya distribusi kelas tetap seimbang.

    Return: dict group_key -> split_name
    """
    rng = random.Random(seed)

    # Kumpulkan komposisi kelas tiap grup
    group_classes: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        group_classes[r["group"]][r["class"]] += 1

    # Kelompokkan grup berdasarkan kelas dominannya
    by_dominant: dict[str, list[str]] = defaultdict(list)
    for g, counter in group_classes.items():
        dominant = counter.most_common(1)[0][0]
        by_dominant[dominant].append(g)

    assignment: dict[str, str] = {}
    for dominant, groups in by_dominant.items():
        groups = sorted(groups)         # deterministik sebelum shuffle
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))

        # Jaminan minimal: kalau grup cukup banyak, val & test tidak boleh kosong
        if n >= 10:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))

        for i, g in enumerate(groups):
            if i < n_test:
                assignment[g] = "test"
            elif i < n_test + n_val:
                assignment[g] = "val"
            else:
                assignment[g] = "train"

    return assignment


def verify_no_leakage(records: list, assignment: dict[str, str],
                      threshold: int = 2) -> list[str]:
    """
    Verifikasi akhir: pastikan tidak ada gambar train yang nyaris identik
    dengan gambar val/test. Ini jaring pengaman kedua setelah dedup.
    Return list peringatan.
    """
    warnings = []
    train_hashes = []
    eval_items = []

    for r in records:
        split = assignment.get(r["group"], "train")
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h = phash_64(img)
        if split == "train":
            train_hashes.append(h)
        else:
            eval_items.append((r, h, split))

    train_arr = train_hashes
    for r, h, split in eval_items:
        for th in train_arr:
            if hamming(h, th) <= threshold:
                warnings.append(
                    f"BOCOR: {Path(r['path']).name} ({split}) mirip gambar train"
                )
                break
    return warnings


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Merge dataset deteksi -> dataset klasifikasi (group-aware, dedup)"
    )
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="Folder dataset Roboflow YOLO (punya train/valid/test)")
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="Folder foto manual berformat <dir>/<nominal>/*.jpg")
    ap.add_argument("--output", default="data/classification")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.10)
    ap.add_argument("--padding", type=float, default=0.06,
                    help="Padding relatif dasar di sekitar bbox (adaptif)")
    ap.add_argument("--min-crop-size", type=int, default=48,
                    help="Sisi terpendek minimum crop (px)")
    ap.add_argument("--min-blur", type=float, default=25.0,
                    help="Ambang variance of Laplacian; 0 = matikan filter blur")
    ap.add_argument("--max-side", type=int, default=640,
                    help="Batas sisi terpanjang crop yang disimpan")
    ap.add_argument("--dedup-threshold", type=int, default=4,
                    help="Jarak Hamming pHash untuk near-duplicate; -1 = matikan")
    ap.add_argument("--verify-leakage", action="store_true",
                    help="Jalankan verifikasi kebocoran train/eval (lambat)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true",
                    help="Hapus folder output kalau sudah ada")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    if output_dir.exists():
        if args.clean:
            shutil.rmtree(output_dir)
            print(f"Folder lama dihapus: {output_dir}")
        else:
            print(f"Peringatan: '{output_dir}' sudah ada. "
                  f"Pakai --clean kalau mau mulai bersih.")
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = output_dir / "_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    for cls in CLASS_ORDER:
        (staging_dir / cls).mkdir(parents=True, exist_ok=True)

    deduper = PhashDeduper(threshold=args.dedup_threshold)
    records: list = []
    stats: Counter = Counter()

    # ── Kumpulkan ──
    for ds_path in args.datasets:
        ds_dir = Path(ds_path).expanduser()
        if not ds_dir.exists():
            print(f"Dataset tidak ditemukan, dilewati: {ds_dir}")
            continue
        print(f"\nMemproses dataset deteksi: {ds_dir.name}")
        collect_from_detection_dataset(
            ds_dir, staging_dir, deduper, args.padding, args.min_crop_size,
            args.min_blur, args.max_side, records, stats,
        )

    for extra_path in args.extra_dirs:
        ex_dir = Path(extra_path).expanduser()
        if not ex_dir.exists():
            print(f"Folder manual tidak ditemukan, dilewati: {ex_dir}")
            continue
        print(f"\nMemproses folder manual: {ex_dir.name}")
        collect_from_class_folders(
            ex_dir, staging_dir, deduper, args.min_crop_size,
            args.min_blur, args.max_side, records, stats,
        )

    if not records:
        print("\nTidak ada crop yang dihasilkan. Cek path dataset kamu.")
        shutil.rmtree(staging_dir, ignore_errors=True)
        sys.exit(1)

    # ── Ringkasan pengumpulan ──
    print("\n" + "=" * 62)
    print("  RINGKASAN PENGUMPULAN")
    print("=" * 62)
    print(f"  Crop disimpan       : {stats['kept']}")
    print(f"  Dibuang - duplikat  : {stats['duplicate']}")
    print(f"  Dibuang - blur      : {stats['too_blurry']}")
    print(f"  Dibuang - kekecilan : {stats['too_small']}")
    print(f"  Dibuang - kelas tak dikenal : {stats['unknown_class']}")
    print(f"  Sumber tanpa label  : {stats['no_label']}")
    print(f"  Sumber tidak terbaca: {stats['unreadable']}")

    n_groups = len({r["group"] for r in records})
    print(f"\n  Total grup (foto sumber unik): {n_groups}")
    print(f"  Rata-rata crop per grup      : {len(records) / n_groups:.2f}")

    print("\n  Distribusi kelas (sebelum split):")
    cls_counter = Counter(r["class"] for r in records)
    for cls in CLASS_ORDER:
        print(f"    Rp {int(cls):>7,} : {cls_counter.get(cls, 0):>6}")

    # ── Split per grup ──
    print(f"\nSplit per GRUP (anti-leakage): "
          f"train={1 - args.val_split - args.test_split:.0%}, "
          f"val={args.val_split:.0%}, test={args.test_split:.0%}")
    assignment = group_stratified_split(records, args.val_split,
                                        args.test_split, args.seed)

    # ── Pindahkan file ke folder final ──
    for split in ("train", "val", "test"):
        for cls in CLASS_ORDER:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    counters = {cls: Counter() for cls in CLASS_ORDER}
    manifest = []
    for r in records:
        split = assignment.get(r["group"], "train")
        cls = r["class"]
        idx = counters[cls][split]
        dst = output_dir / split / cls / f"{cls}_{split}_{idx:05d}.jpg"
        shutil.move(r["path"], dst)
        counters[cls][split] += 1
        manifest.append({
            "file": str(dst.relative_to(output_dir)),
            "class": cls,
            "split": split,
            "group": r["group"],
            "origin": r["origin"],
            "source": r["source"],
            "h": r["h"],
            "w": r["w"],
        })

    shutil.rmtree(staging_dir, ignore_errors=True)

    # ── Verifikasi kebocoran (opsional) ──
    if args.verify_leakage:
        print("\nMenjalankan verifikasi kebocoran train/eval...")
        recs_final = [
            {"path": str(output_dir / m["file"]), "group": m["group"]}
            for m in manifest
        ]
        assign_final = {m["group"]: m["split"] for m in manifest}
        warns = verify_no_leakage(recs_final, assign_final, threshold=2)
        if warns:
            print(f"  Ditemukan {len(warns)} indikasi kebocoran:")
            for w in warns[:20]:
                print(f"    {w}")
            if len(warns) > 20:
                print(f"    ... dan {len(warns) - 20} lainnya")
            print("  Saran: naikkan --dedup-threshold lalu jalankan ulang.")
        else:
            print("  Bersih, tidak ada kebocoran terdeteksi.")

    # ── Tabel hasil ──
    print("\n" + "=" * 62)
    print("  DISTRIBUSI AKHIR")
    print("=" * 62)
    print(f"  {'Kelas':>10} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
    print(f"  {'-' * 46}")
    totals = Counter()
    for cls in CLASS_ORDER:
        c = counters[cls]
        tot = c["train"] + c["val"] + c["test"]
        totals["train"] += c["train"]
        totals["val"] += c["val"]
        totals["test"] += c["test"]
        print(f"  {cls:>10} {c['train']:>8} {c['val']:>8} {c['test']:>8} {tot:>8}")
    print(f"  {'-' * 46}")
    grand = totals["train"] + totals["val"] + totals["test"]
    print(f"  {'TOTAL':>10} {totals['train']:>8} {totals['val']:>8} "
          f"{totals['test']:>8} {grand:>8}")

    # ── Simpan metadata ──
    (output_dir / "labels.txt").write_text(
        "\n".join(CLASS_ORDER) + "\n", encoding="utf-8"
    )
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_images": grand,
            "n_groups": n_groups,
            "split_counts": dict(totals),
            "class_counts": {c: dict(counters[c]) for c in CLASS_ORDER},
            "collection_stats": dict(stats),
            "args": vars(args),
        }, f, indent=2)

    print(f"\nDataset tersimpan di: {output_dir.resolve()}")
    print("Lanjut: python scripts/00b_preflight_check.py --data "
          f"{args.output}")


if __name__ == "__main__":
    main()
