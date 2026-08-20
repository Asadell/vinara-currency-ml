"""
src/augment.py
==============
Pipeline augmentasi khusus untuk simulasi kondisi uang dunia nyata.

Filosofi: model sekarang 99.5% di uang bersih tapi jeblok di uang lecek.
Itu artinya train-set kita cuma mewakili SATU domain (uang rapi, pencahayaan
studio). Solusinya domain randomization: paksa model lihat variasi degradasi
fisik + variasi pencahayaan waktu training, supaya distribusi train melebar
sampai menutupi distribusi dunia nyata.

Kelompok augmentasi (tiap kelompok punya alasan fisik):

  A. DEFORMASI GEOMETRIS (uang lecek, terlipat, dipegang, tidak rata)
     - ElasticTransform      -> kerutan halus kertas lusuh
     - GridDistortion        -> gelombang/lipatan besar
     - Perspective           -> uang dipegang miring, kamera tidak tegak lurus
     - Affine rotate/shear   -> orientasi bebas
     - CreaseShadow (custom) -> garis lipatan + bayangan di sepanjang lipatan

  B. DEGRADASI PERMUKAAN (coretan, stempel, sobek, kotor)
     - CoarseDropout         -> sobek/lubang/stiker
     - RandomScribble (custom) -> coretan pulpen & stempel
     - GaussNoise / ISONoise -> sensor HP murah

  C. PENCAHAYAAN (warung remang, lampu kuning, backlight, bayangan tangan)
     - RandomBrightnessContrast
     - RandomGamma           -> gelap/terang non-linear
     - ColorJitter/HueSat    -> lampu kuning/putih/neon
     - RandomShadow          -> bayangan tangan/badan
     - RandomToneCurve

  D. DEGRADASI OPTIK (HP mid-low)
     - MotionBlur / Defocus  -> tangan goyang, autofocus gagal
     - ImageCompression      -> artefak JPEG dari kamera murah
     - Downscale             -> sensor resolusi rendah

Penting soal HUE: kode lama sengaja TIDAK pakai hue shift karena "warna itu
fitur uang". Itu keliru arah. Justru karena model kelewat bergantung warna,
dia bingung waktu lampu kuning warung menggeser semua warna. Yang benar:
hue shift TERBATAS (+/- 8 derajat) supaya model belajar angka & pola juga,
bukan cuma histogram warna. Jangan digeser ekstrem (+/- 50) karena itu
memang bisa bikin 2rb abu ketuker 20rb hijau.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except ImportError:  # pragma: no cover
    A = None
    _HAS_ALBUMENTATIONS = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom transform: simulasi lipatan uang
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_crease(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulasi bekas lipatan uang: garis terang/gelap yang membelah uang,
    plus gradien bayangan di kedua sisi garis.

    Uang yang sering dilipat punya bekas lipatan vertikal di tengah
    (lipat dua) atau beberapa garis (lipat empat/dompet).
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    n_creases = int(rng.integers(1, 4))
    for _ in range(n_creases):
        vertical = rng.random() < 0.7  # lipatan uang lebih sering vertikal
        length = w if vertical else h
        pos = int(rng.integers(int(length * 0.2), int(length * 0.8)))
        width_px = max(2, int(rng.integers(2, max(3, length // 40))))
        strength = float(rng.uniform(0.10, 0.30))

        # Bikin mask gradien di sekitar garis lipatan
        coords = np.arange(length, dtype=np.float32)
        falloff = np.exp(-((coords - pos) ** 2) / (2 * (width_px * 3.0) ** 2))

        # Satu sisi lebih gelap, sisi lain lebih terang (efek 3D lipatan)
        sign = np.where(coords < pos, -1.0, 1.0).astype(np.float32)
        profile = 1.0 + sign * falloff * strength

        if vertical:
            out *= profile[None, :, None]
        else:
            out *= profile[:, None, None]

        # Garis lipatan itu sendiri (serat kertas rusak, sedikit lebih terang)
        line_strength = float(rng.uniform(1.02, 1.12))
        if vertical:
            x1 = max(0, pos - width_px // 2)
            x2 = min(w, pos + width_px // 2 + 1)
            out[:, x1:x2] *= line_strength
        else:
            y1 = max(0, pos - width_px // 2)
            y2 = min(h, pos + width_px // 2 + 1)
            out[y1:y2, :] *= line_strength

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_scribble(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulasi coretan pulpen / stempel / tanda tangan di uang.
    Ini umum banget di uang beredar Indonesia.
    """
    h, w = img.shape[:2]
    out = img.copy()

    mode = rng.random()

    if mode < 0.55:
        # Coretan pulpen: polyline acak
        n_strokes = int(rng.integers(1, 4))
        for _ in range(n_strokes):
            n_pts = int(rng.integers(3, 7))
            pts = np.stack([
                rng.integers(0, w, size=n_pts),
                rng.integers(0, h, size=n_pts),
            ], axis=1).astype(np.int32)
            color = tuple(int(c) for c in rng.integers(0, 90, size=3))
            thickness = int(rng.integers(1, max(2, h // 60)))
            cv2.polylines(out, [pts], isClosed=False, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)

    elif mode < 0.85:
        # Stempel bulat semi-transparan
        overlay = out.copy()
        cx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
        cy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
        radius = int(rng.integers(max(6, h // 8), max(8, h // 3)))
        color = tuple(int(c) for c in rng.integers(0, 130, size=3))
        cv2.circle(overlay, (cx, cy), radius, color,
                   thickness=max(2, radius // 8), lineType=cv2.LINE_AA)
        cv2.line(overlay, (cx - radius // 2, cy), (cx + radius // 2, cy),
                 color, thickness=max(1, radius // 10), lineType=cv2.LINE_AA)
        alpha = float(rng.uniform(0.35, 0.75))
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    else:
        # Noda / kotoran: blob gelap blur
        overlay = np.zeros((h, w), dtype=np.uint8)
        n_blobs = int(rng.integers(1, 4))
        for _ in range(n_blobs):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            ax = int(rng.integers(max(3, w // 25), max(5, w // 8)))
            ay = int(rng.integers(max(3, h // 25), max(5, h // 8)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(overlay, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(1.0, w / 60.0))
        mask = (overlay.astype(np.float32) / 255.0)[..., None]
        darkness = float(rng.uniform(0.25, 0.55))
        out = (out.astype(np.float32) * (1.0 - mask * darkness))
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def _apply_crumple(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulasi uang kusut (crumpled) pakai displacement map low-frequency
    + shading berbasis gradien displacement.

    Cara kerja: bikin noise low-frequency, blur, pakai sebagai peta pergeseran
    pixel. Gradien peta itu dipakai buat shading supaya kelihatan ada tonjolan
    dan cekungan, bukan cuma distorsi datar.
    """
    h, w = img.shape[:2]
    scale = float(rng.uniform(3.0, 9.0))          # kekuatan pergeseran (px)
    smooth = float(rng.uniform(w / 22.0, w / 9.0))  # kehalusan kerutan

    noise_x = rng.random((h, w)).astype(np.float32) - 0.5
    noise_y = rng.random((h, w)).astype(np.float32) - 0.5
    dx = cv2.GaussianBlur(noise_x, (0, 0), sigmaX=smooth, sigmaY=smooth)
    dy = cv2.GaussianBlur(noise_y, (0, 0), sigmaX=smooth, sigmaY=smooth)

    # Normalisasi supaya amplitudo konsisten
    for d in (dx, dy):
        m = np.abs(d).max()
        if m > 1e-6:
            d /= m

    dx *= scale
    dy *= scale

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x + dx, 0, w - 1)
    map_y = np.clip(grid_y + dy, 0, h - 1)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    # Shading: gradien displacement -> tonjolan terang, cekungan gelap
    gx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
    shade = gx + gy
    m = np.abs(shade).max()
    if m > 1e-6:
        shade /= m
    intensity = float(rng.uniform(0.10, 0.28))
    shade = 1.0 + shade * intensity

    out = warped.astype(np.float32) * shade[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


class CurrencyWear(A.ImageOnlyTransform if _HAS_ALBUMENTATIONS else object):
    """
    Transform gabungan: lipatan + kusut + coretan.
    Dipisah dari transform Albumentations bawaan karena tiga efek ini
    tidak ada padanannya dan paling relevan buat uang beredar.
    """

    def __init__(self, crease_p=0.5, crumple_p=0.5, scribble_p=0.35,
                 always_apply=False, p=0.7):
        if _HAS_ALBUMENTATIONS:
            super().__init__(p=p)
        self.crease_p = crease_p
        self.crumple_p = crumple_p
        self.scribble_p = scribble_p

    def apply(self, img, **params):
        rng = np.random.default_rng()
        out = img
        if rng.random() < self.crumple_p:
            out = _apply_crumple(out, rng)
        if rng.random() < self.crease_p:
            out = _apply_crease(out, rng)
        if rng.random() < self.scribble_p:
            out = _apply_scribble(out, rng)
        return out

    def get_transform_init_args_names(self):
        return ("crease_p", "crumple_p", "scribble_p")


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline utama
# ═══════════════════════════════════════════════════════════════════════════════

def build_train_transform(strength: str = "medium"):
    """
    Bangun pipeline Albumentations untuk training.

    Args:
        strength: "light" | "medium" | "heavy"
            light  -> buat sanity check / dataset sudah beragam
            medium -> DEFAULT, rekomendasi untuk kasus GUIDIO
            heavy  -> kalau train-set kamu benar-benar cuma uang studio bersih

    Returns:
        albumentations.Compose yang menerima & mengembalikan uint8 HWC RGB
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations belum terinstall. Jalankan: pip install albumentations"
        )

    presets = {
        "light":  dict(geo=0.35, wear=0.35, light=0.45, optic=0.30),
        "medium": dict(geo=0.60, wear=0.65, light=0.75, optic=0.55),
        "heavy":  dict(geo=0.80, wear=0.85, light=0.90, optic=0.75),
    }
    if strength not in presets:
        raise ValueError(f"strength harus salah satu dari {list(presets)}")
    p = presets[strength]

    return A.Compose([
        # ── A. Orientasi dasar ────────────────────────────────────────────────
        # CATATAN: JANGAN pakai HorizontalFlip. Uang yang dicerminkan
        # menghasilkan tulisan terbalik yang TIDAK PERNAH ada di dunia nyata.
        # Itu cuma buang kapasitas model. Yang realistis adalah rotasi 180
        # (uang kebalik) dan rotasi kecil (uang miring di tangan).
        A.RandomRotate90(p=0.0),  # dimatikan: uang jarang tegak 90 derajat
        A.Affine(
            rotate=(-180, 180),
            scale=(0.85, 1.15),
            shear={"x": (-8, 8), "y": (-8, 8)},
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.85,
        ),

        # ── B. Deformasi (lecek, terlipat, tidak rata) ────────────────────────
        A.Perspective(scale=(0.03, 0.14), keep_size=True,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, p=p["geo"]),
        A.OneOf([
            A.ElasticTransform(alpha=40, sigma=8, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.28, p=1.0),
            A.OpticalDistortion(distort_limit=0.25, p=1.0),
        ], p=p["geo"]),

        # Lipatan + kusut + coretan (custom)
        CurrencyWear(crease_p=0.55, crumple_p=0.55, scribble_p=0.30,
                     p=p["wear"]),

        # Sobek / stiker / tertutup jari
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.04, 0.16),
            hole_width_range=(0.04, 0.16),
            fill=0,
            p=0.30,
        ),

        # ── C. Pencahayaan ────────────────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=(-0.38, 0.30),
                                   contrast_limit=(-0.30, 0.30),
                                   p=p["light"]),
        A.RandomGamma(gamma_limit=(55, 145), p=p["light"] * 0.8),
        # Hue TERBATAS: cukup buat simulasi lampu kuning warung / neon,
        # tapi tidak sampai bikin 2rb (abu) ketuker 20rb (hijau).
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=28,
                             val_shift_limit=22, p=p["light"] * 0.8),
        A.RandomToneCurve(scale=0.22, p=0.30),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3),
                       shadow_dimension=5, p=0.35),

        # ── D. Degradasi optik HP mid-low ─────────────────────────────────────
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(1, 4), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=p["optic"]),
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=p["optic"] * 0.8),
        A.Downscale(scale_range=(0.35, 0.75),
                    interpolation_pair={"downscale": cv2.INTER_AREA,
                                        "upscale": cv2.INTER_LINEAR},
                    p=0.28),
        A.ImageCompression(quality_range=(30, 88), p=0.45),
    ], p=1.0)


def build_eval_transform():
    """
    Transform untuk val/test: TIDAK ada augmentasi sama sekali.
    Resize/letterbox ditangani di luar (tf.data), jadi ini identity.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")
    return A.Compose([])


def build_hard_eval_transform(seed_offset: int = 0):
    """
    Transform untuk membentuk "hard test set" secara sintetis:
    uang bersih -> disimulasikan jadi lecek + gelap.

    Ini BUKAN pengganti foto uang lecek asli, tapi berguna sebagai
    regression test cepat: kalau akurasi di sini anjlok drastis,
    berarti model masih rapuh.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")
    return A.Compose([
        A.Perspective(scale=(0.06, 0.14), keep_size=True,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.8),
        CurrencyWear(crease_p=0.8, crumple_p=0.8, scribble_p=0.4, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.40, -0.10),
                                   contrast_limit=(-0.25, 0.0), p=0.9),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=-20,
                             val_shift_limit=-15, p=0.7),
        A.MotionBlur(blur_limit=(5, 11), p=0.6),
        A.ImageCompression(quality_range=(25, 55), p=0.7),
    ], p=1.0)
