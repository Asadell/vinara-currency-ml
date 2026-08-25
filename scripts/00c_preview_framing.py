#!/usr/bin/env python3
"""
00c_preview_framing.py
======================
Dump grid PNG berisi contoh sampel training SETELAH simulasi framing kamera,
persis seperti yang akan dilihat model (224x224, sudah letterbox).

JALANKAN INI DULU sebelum training. Dua alasan:

  1. Augmentasi yang salah tidak pernah melempar error - dia cuma bikin
     akurasi turun diam-diam. Satu-satunya cara tahu adalah melihatnya.
  2. Signature Albumentations berubah antar versi (fill/fill_mask/border_mode).
     Script ini akan gagal keras di sini, bukan setelah 50 epoch.

Usage:
    python scripts/00c_preview_framing.py \
        --data data/classification \
        --bg-dir ~/backgrounds \
        --out preview_framing.png \
        --rows 6 --cols 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.augment import (  # noqa: E402
    build_geometric_transform,
    build_photometric_transform,
)
from src.common import CLASS_ORDER  # noqa: E402
from src.data import scan_split_per_class  # noqa: E402
from src.framing import SceneFramer  # noqa: E402


def letterbox_224(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Replika persis `tf.image.resize_with_pad` + pad 0."""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh,
           (size - nw) // 2:(size - nw) // 2 + nw] = cv2.resize(
        img, (nw, nh), interpolation=interp)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/classification")
    ap.add_argument("--split", default="train")
    ap.add_argument("--bg-dir", default=None)
    ap.add_argument("--out", default="preview_framing.png")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--aug-strength", default="medium",
                    choices=["light", "medium", "heavy"])
    ap.add_argument("--frame-prob", type=float, default=0.75)
    ap.add_argument("--frame-scale-min", type=float, default=0.28)
    ap.add_argument("--frame-scale-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per_class = scan_split_per_class(Path(args.data), args.split)
    pool = [(p, CLASS_ORDER[i]) for i, ps in enumerate(per_class) for p in ps]
    if not pool:
        raise SystemExit(f"Split '{args.split}' kosong di {args.data}")
    print(f"{len(pool)} gambar di split '{args.split}'")

    geo = build_geometric_transform(args.aug_strength)
    photo = build_photometric_transform(args.aug_strength)
    framer = SceneFramer(bg_dir=args.bg_dir,
                         scale_range=(args.frame_scale_min, args.frame_scale_max))
    print(f"Latar: {len(framer.pool)} foto"
          if len(framer.pool) else "Latar: PROSEDURAL (tidak ada --bg-dir)")

    rng = np.random.default_rng(args.seed)
    tiles, labels = [], []

    for _ in range(args.rows * args.cols):
        path, cls = pool[int(rng.integers(0, len(pool)))]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        padded, mask = framer.prepare(img)
        out = geo(image=padded, mask=mask)
        scale = ((args.frame_scale_min, args.frame_scale_max)
                 if rng.random() < args.frame_prob else (0.90, 1.0))
        scene = framer.compose(out["image"], out["mask"], rng, scale_range=scale)
        scene = photo(image=scene)["image"]

        tile = letterbox_224(scene)
        cv2.putText(tile, cls, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(tile)
        labels.append(cls)

    n = args.rows * args.cols
    while len(tiles) < n:
        tiles.append(np.zeros((224, 224, 3), np.uint8))

    grid = np.vstack([
        np.hstack(tiles[r * args.cols:(r + 1) * args.cols])
        for r in range(args.rows)
    ])
    cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Tersimpan: {args.out}  ({grid.shape[1]}x{grid.shape[0]})")
    print("\nYang harus kamu lihat:")
    print("  - Uang TIDAK terpotong ujungnya walau diputar 90/180 derajat")
    print("  - Uang menempel di latar tanpa segi empat hitam mengelilinginya")
    print("  - Ada campuran: uang penuh sebidang DAN uang kecil di frame tegak")
    print("  - Angka nominal masih terbaca di sebagian besar sampel besar")


if __name__ == "__main__":
    main()
