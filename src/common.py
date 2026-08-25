"""
src/common.py
=============
Helper bersama untuk seluruh pipeline rupiah-vision.

Isi:
  - Definisi kelas & mapping
  - Letterbox resize (JAGA aspect ratio, penting untuk Rupiah)
  - Scanner file gambar
  - Perceptual hash untuk deduplikasi

CATATAN ASPECT RATIO (penting):
    Ukuran fisik uang Rupiah emisi 2016/2022 (mm):
        1.000   -> 121 x 65  (rasio 1.86)
        2.000   -> 126 x 65  (rasio 1.94)
        5.000   -> 131 x 65  (rasio 2.02)
        10.000  -> 136 x 65  (rasio 2.09)
        20.000  -> 141 x 65  (rasio 2.17)
        50.000  -> 146 x 65  (rasio 2.25)
        100.000 -> 151 x 65  (rasio 2.32)

    Rasio panjang:lebar naik monoton seiring nominal. Ini sinyal
    diskriminatif GRATIS. Kalau crop di-resize paksa ke 224x224
    (squash), sinyal ini HANCUR total. Makanya kita pakai
    letterbox (resize + pad) yang menjaga rasio asli.

    Catatan: rasio ini cuma valid kalau uang relatif rata. Buat uang
    terlipat rasionya kacau, jadi ini sinyal TAMBAHAN, bukan satu-satunya.
    Model tetap harus belajar warna + angka + pola.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ─── Kelas ─────────────────────────────────────────────────────────────────────

CLASS_ORDER = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
NUM_CLASSES = len(CLASS_ORDER)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_ORDER)}

# Label yang enak dibaca TTS
CLASS_TO_SPOKEN = {
    "1000": "seribu rupiah",
    "2000": "dua ribu rupiah",
    "5000": "lima ribu rupiah",
    "10000": "sepuluh ribu rupiah",
    "20000": "dua puluh ribu rupiah",
    "50000": "lima puluh ribu rupiah",
    "100000": "seratus ribu rupiah",
}

# Rasio aspek nominal (panjang / lebar) dari ukuran fisik resmi BI
NOMINAL_ASPECT = {
    "1000": 121 / 65,
    "2000": 126 / 65,
    "5000": 131 / 65,
    "10000": 136 / 65,
    "20000": 141 / 65,
    "50000": 146 / 65,
    "100000": 151 / 65,
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Mapping index kelas Roboflow -> nama folder kita.
# Diverifikasi dari data.yaml ketiga dataset (urutan alfabetis Bahasa Indonesia).
ROBOFLOW_IDX_TO_CLASS = {
    0: "20000",   # dua puluh ribu
    1: "2000",    # dua ribu
    2: "50000",   # lima puluh ribu
    3: "5000",    # lima ribu
    4: "10000",   # sepuluh ribu
    5: "100000",  # seratus ribu
    6: "1000",    # seribu
}


# ─── Letterbox ─────────────────────────────────────────────────────────────────

def letterbox_numpy(
    img: np.ndarray,
    size: int = 224,
    pad_value: int = 0,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize gambar ke size x size TANPA merusak aspect ratio.
    Sisi kosong diisi pad_value (default hitam = 0).

    Args:
        img: array HWC uint8 (BGR atau RGB, bebas)
        size: sisi output persegi
        pad_value: nilai isian padding
    Returns:
        array (size, size, 3) uint8
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad_value, dtype=np.uint8)

    scale = min(size / h, size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = interpolation if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


# ─── Scanner ───────────────────────────────────────────────────────────────────

def list_images(folder: Path) -> list[Path]:
    """List semua file gambar di satu folder (non-rekursif), terurut."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def list_images_recursive(folder: Path) -> list[Path]:
    """List semua file gambar rekursif, terurut."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


# ─── Perceptual hash (dedup) ───────────────────────────────────────────────────

def phash_64(img: np.ndarray) -> int:
    """
    Perceptual hash 64-bit (DCT-based, setara imagehash.phash).
    Diimplementasi manual pakai OpenCV supaya tidak wajib install Pillow+imagehash,
    tapi kalau paket `imagehash` tersedia hasilnya sebanding.

    Returns:
        integer 64-bit
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    dct_low = dct[:8, :8]

    # Buang komponen DC (rata-rata) supaya tidak didominasi brightness
    med = np.median(dct_low[1:, 1:])

    bits = (dct_low > med).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    """Jarak Hamming antara dua hash integer."""
    return bin(a ^ b).count("1")


def _hash_bands(h: int, n_bands: int) -> list[tuple[int, int]]:
    """Pecah hash 64-bit jadi n_bands potongan. Return [(band_id, nilai), ...]."""
    out = []
    start = 0
    for b in range(n_bands):
        width = (64 - start) // (n_bands - b)
        val = (h >> start) & ((1 << width) - 1)
        out.append((b, val))
        start += width
    return out


def dedup_by_phash(
    items: list,
    hash_fn,
    threshold: int = 4,
) -> tuple[list, int]:
    """
    Buang near-duplicate dari list item berbasis perceptual hash.

    PERBAIKAN BUG (versi lama bocor sekitar 70%):

    Versi lama mem-bucket dengan `h >> 48`, yaitu 16 bit TERATAS saja, lalu
    hanya membandingkan di dalam bucket yang sama. Untuk threshold=4, peluang
    keempat bit yang berbeda kebetulan semuanya jatuh di 48 bit bawah adalah

        C(48,4) / C(64,4) = 4.669.920 / 15.249.024 = 0,306

    jadi sekitar 69% near-duplicate TIDAK PERNAH dibandingkan dan lolos.
    Komentar di versi lama juga menjanjikan "fallback cek linear kalau bucket
    kosong", tapi kodenya tidak pernah melakukan itu.

    Ini serius karena seluruh premis pipeline ini adalah "akurasi 99,5% kamu
    palsu gara-gara kebocoran". Dedup yang bocor membuat angka evaluasi tetap
    tidak jujur, dengan cara yang tidak kelihatan.

    Versi baru pakai multi-index LSH dan EKSAK untuk jarak <= threshold:
    hash dipecah jadi (threshold + 1) band. Kalau dua hash berbeda paling
    banyak `threshold` bit, menurut pigeonhole setidaknya SATU band pasti
    identik, jadi pasangan itu dijamin masuk daftar kandidat. Kandidat lalu
    diverifikasi dengan jarak Hamming sebenarnya.

    Args:
        items: list item apa pun
        hash_fn: fungsi item -> int hash 64-bit
        threshold: jarak Hamming maksimum yang dianggap duplikat
                   (0 = identik persis, 4-6 = near-duplicate, >10 = beda,
                   negatif = matikan dedup)
    Returns:
        (items_unik, jumlah_dibuang)
    """
    if threshold < 0:
        return list(items), 0

    n_bands = min(64, max(1, threshold + 1))
    buckets: dict[tuple[int, int], list[int]] = {}
    kept = []
    removed = 0

    for item in items:
        h = hash_fn(item)
        bands = _hash_bands(h, n_bands)

        is_dup = False
        seen: set[int] = set()
        for key in bands:
            for other in buckets.get(key, ()):
                if other in seen:
                    continue
                seen.add(other)
                if hamming(h, other) <= threshold:
                    is_dup = True
                    break
            if is_dup:
                break

        if is_dup:
            removed += 1
            continue

        for key in bands:
            buckets.setdefault(key, []).append(h)
        kept.append(item)

    return kept, removed


# ─── Statistik gambar ──────────────────────────────────────────────────────────

def blur_score(img: np.ndarray) -> float:
    """
    Variance of Laplacian. Makin kecil makin blur.
    Dipakai di preflight check untuk mendeteksi crop yang tidak layak latih.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_stats(img: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) brightness pada channel V (HSV)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else img
    v = hsv[..., 2] if hsv.ndim == 3 else hsv
    return float(v.mean()), float(v.std())
