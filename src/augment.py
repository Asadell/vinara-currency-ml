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

  E. ANTI JALAN-PINTAS WARNA (build_decolor_group)
     - ToGray, desaturasi kuat, hue lebar

REVISI SOAL HUE (penting, ini mengoreksi versi sebelumnya):

Versi sebelumnya membatasi hue di +/-8 derajat dengan alasan "jangan sampai
2rb abu ketuker 20rb hijau". Setelah diukur, batasan itu ternyata JUSTRU
mempertahankan masalahnya. Model hasil pipeline lama mendapat 80.5% pada
gambar normal tapi cuma 15.2% saat warnanya dibuang, padahal tebak acak
untuk 7 kelas adalah 14.3%. Dengan kata lain model itu 100% classifier
warna dan tidak pernah membaca angka nominal sama sekali.

Selama warna selalu cukup untuk menjawab, model tidak punya alasan belajar
yang lain. Maka sekarang sebagian sampel sengaja dibuat MUSTAHIL dijawab
pakai warna (lihat build_decolor_group). Hue +/-8 tetap dipakai di jalur
pencahayaan biasa; pergeseran yang lebih ekstrem hanya muncul di kelompok
decolor dengan probabilitas terbatas.
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

# Catatan kalibrasi frekuensi (diukur, bukan ditebak):
#   crop uang dari HP kelas bawah  -> variance of Laplacian ~19
#   crop dataset asli              -> ~813
#   dataset + GaussianBlur k7 + Downscale 0.35 + JPEG q30 -> ~15
# Jadi RENTANG augmentasi sudah menjangkau HP burem, yang kurang FREKUENSInya:
# di preset lama peluang ketiganya menyala bersamaan cuma
# 0.55 x 0.28 x 0.45 = 6.9% sampel. `optic` dan `Downscale` dinaikkan.
#
# `decolor` adalah kelompok BARU, lihat build_decolor_group().
_PRESETS = {
    "light":  dict(geo=0.35, wear=0.35, light=0.45, optic=0.45, decolor=0.15),
    "medium": dict(geo=0.60, wear=0.65, light=0.75, optic=0.75, decolor=0.28),
    "heavy":  dict(geo=0.80, wear=0.85, light=0.90, optic=0.85, decolor=0.38),
}


def _preset(strength: str) -> dict:
    if not _HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations belum terinstall. Jalankan: pip install albumentations"
        )
    if strength not in _PRESETS:
        raise ValueError(f"strength harus salah satu dari {list(_PRESETS)}")
    return _PRESETS[strength]


def build_decolor_group(strength: str = "medium"):
    """
    KELOMPOK BARU: paksa model berhenti mengandalkan warna.

    KENAPA INI ADA
    --------------
    Model lama diukur begini pada 210 gambar test bersih:

        gambar apa adanya           -> akurasi 80.5%
        warna dibuang (grayscale)   -> akurasi 15.2%   (tebak acak 7 kelas = 14.3%)
        warna utuh, detail 12 px    -> akurasi 56.7%

    Artinya model TIDAK PERNAH membaca angka nominalnya. Dia cuma mencocokkan
    histogram warna. Itu bekerja 80% di dataset karena tiap nominal punya warna
    dominan yang berbeda, tapi runtuh persis di 20rb (hijau) vs 50rb (biru)
    saat cahaya redup membuat dua warna itu berdekatan.

    Bukti tambahan: akurasi nyaris tidak berubah dari 224 px (80.5%) ke 63 px
    (78.6%). Kalau model benar-benar membaca teks "20000", turun ke 63 px pasti
    menghancurkannya. Datar = dia memang tidak melihat teks.

    Penyebabnya ada di kode lama, tertulis eksplisit di docstring modul ini:
    hue sengaja dibatasi +/-8 supaya "2rb abu tidak ketuker 20rb hijau". Niatnya
    benar, tapi efeknya melindungi jalan pintas itu. Model tidak pernah
    dihadapkan ke sampel yang MUSTAHIL dijawab pakai warna, jadi dia tidak
    pernah terpaksa belajar angka dan pola.

    Kelompok ini menyediakan sampel seperti itu:
      - ToGray            : nol informasi warna, satu-satunya jalan adalah pola
      - desaturasi kuat   : uang pudar / cahaya redup, warna nyaris hilang
      - hue lebar         : warna ada tapi BOHONG, jadi tidak bisa dipercaya

    EKSPEKTASI: akurasi training akan TURUN di epoch-epoch awal dibanding
    sebelumnya. Itu tandanya bekerja, bukan tandanya rusak. Yang harus naik
    adalah metrik COLOR-STRESS di 01_train.py (akurasi pada test grayscale).
    Target realistis: dari 15% naik ke setidaknya 55-65%.
    """
    p = _preset(strength)

    return A.OneOf([
        # Nol warna. Sampel ini cuma bisa dijawab lewat angka, potret, dan pola.
        A.ToGray(p=1.0),
        # Warna nyaris hilang: uang pudar, lampu remang, sensor HP murah.
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=(-75, -40),
                             val_shift_limit=20, p=1.0),
        # Warna ADA tapi digeser jauh: melatih model tidak percaya warna mentah.
        # Sengaja tidak dipakai sesering dua di atas (bobot OneOf sama rata,
        # jadi masing-masing sekitar sepertiga dari p["decolor"]).
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=35,
                             val_shift_limit=20, p=1.0),
    ], p=p["decolor"])


def build_geometric_transform(strength: str = "medium"):
    """
    TAHAP 2 pipeline: geometri + keausan fisik, dijalankan pada crop uang
    SAJA dan MASK-AWARE (`transform(image=..., mask=...)`).

    Mask-nya penting: setelah rotasi/perspektif kita harus tahu piksel mana
    yang benar-benar uang, supaya `framing.compose_scene()` bisa menempel
    uang ke latar tanpa ikut membawa segi empat hitam sisa padding.

    Dipanggil pada gambar yang SUDAH diberi margin oleh
    `framing.pad_for_rotation()`, jadi rotasi 90/180 derajat tidak lagi
    memotong ujung uang seperti di pipeline lama.
    """
    p = _preset(strength)

    return A.Compose([
        # ── A. Orientasi ──────────────────────────────────────────────────────
        # JANGAN pakai HorizontalFlip: uang bercermin menghasilkan tulisan
        # terbalik yang tidak pernah ada di dunia nyata.
        #
        # RandomRotate90 SEKARANG DINYALAKAN (dulu p=0.0). Alasannya berubah:
        # begitu adegan penuh disimulasikan, uang tegak di dalam frame
        # portrait itu kejadian SANGAT umum - persis kasus fixture yang
        # gagal. Aman dilakukan karena margin rotasi sudah disiapkan.
        A.RandomRotate90(p=0.35),
        A.Affine(
            rotate=(-180, 180),
            scale=(0.85, 1.15),
            shear={"x": (-8, 8), "y": (-8, 8)},
            translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.90,
        ),

        # ── B. Deformasi (lecek, terlipat, tidak rata) ────────────────────────
        A.Perspective(scale=(0.03, 0.14), keep_size=True,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
                      p=p["geo"]),
        A.OneOf([
            A.ElasticTransform(alpha=40, sigma=8,
                               border_mode=cv2.BORDER_CONSTANT,
                               fill=0, fill_mask=0, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.28,
                             border_mode=cv2.BORDER_CONSTANT,
                             fill=0, fill_mask=0, p=1.0),
            A.OpticalDistortion(distort_limit=0.25,
                                border_mode=cv2.BORDER_CONSTANT,
                                fill=0, fill_mask=0, p=1.0),
        ], p=p["geo"]),

        CurrencyWear(crease_p=0.55, crumple_p=0.55, scribble_p=0.30,
                     p=p["wear"]),

        # Sobek / lubang: fill_mask=0 supaya lubangnya benar-benar tembus ke
        # latar waktu compositing, bukan jadi kotak hitam di atas uang.
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.04, 0.16),
            hole_width_range=(0.04, 0.16),
            fill=0,
            fill_mask=0,
            p=0.30,
        ),
    ], p=1.0)


def build_photometric_transform(strength: str = "medium"):
    """
    TAHAP 4 pipeline: pencahayaan + degradasi optik, dijalankan pada
    SELURUH ADEGAN setelah uang ditempel ke latar.

    Urutan ini bukan detail kosmetik. Cahaya ruangan, guncangan tangan,
    dan kompresi JPEG kamera mengenai seluruh foto. Kalau degradasi cuma
    dikenakan ke lembar uang (seperti pipeline lama), uang dan latarnya
    punya karakter noise yang berbeda - dan itu petunjuk gratis yang
    dipakai model untuk "menemukan" uang tanpa benar-benar belajar
    memisahkan objek dari latar.
    """
    p = _preset(strength)

    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.38, 0.30),
                                   contrast_limit=(-0.30, 0.30),
                                   p=p["light"]),
        A.RandomGamma(gamma_limit=(55, 145), p=p["light"] * 0.8),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=28,
                             val_shift_limit=22, p=p["light"] * 0.8),
        A.RandomToneCurve(scale=0.22, p=0.30),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3),
                       shadow_dimension=5, p=0.35),

        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(1, 4), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=p["optic"]),
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=p["optic"] * 0.8),
        # p dinaikkan 0.28 -> 0.50: ini yang paling menentukan apakah sampel
        # sampai ke tingkat ketajaman foto HP kelas bawah.
        A.Downscale(scale_range=(0.30, 0.75),
                    interpolation_pair={"downscale": cv2.INTER_AREA,
                                        "upscale": cv2.INTER_LINEAR},
                    p=0.50),
        A.ImageCompression(quality_range=(25, 88), p=0.55),

        # CLAHE sebagai AUGMENTASI, bukan preprocessing tetap.
        # Diukur: memakai CLAHE sebagai preprocessing tetap saat inferensi
        # justru MENURUNKAN akurasi (80.5% -> 75.2% pada clip=2, 69.5% pada
        # clip=3), karena dia mengganggu relasi warna lewat channel L di LAB.
        # Sebagai augmentasi berprobabilitas kecil dia aman dan berguna: banyak
        # HP menerapkan penajaman lokal sendiri di pipeline kameranya.
        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.15),

        # Anti jalan-pintas warna. Ditaruh PALING AKHIR supaya berlaku pada
        # adegan yang sudah lengkap (uang + latar + cahaya), bukan cuma
        # pada lembar uangnya.
        build_decolor_group(strength),
    ], p=1.0)


def build_train_transform(strength: str = "medium"):
    """
    Pipeline SATU TAHAP versi lama (crop rapat, tanpa simulasi adegan).

    Masih dipakai sebagai jalur `--frame-prob 0`, dan untuk porsi sampel
    yang sengaja dibiarkan berupa crop rapat supaya model tetap jago di
    kasus ideal (uang memenuhi bingkai panduan).
    """
    p = _preset(strength)

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
        A.Downscale(scale_range=(0.30, 0.75),
                    interpolation_pair={"downscale": cv2.INTER_AREA,
                                        "upscale": cv2.INTER_LINEAR},
                    p=0.50),
        A.ImageCompression(quality_range=(25, 88), p=0.55),
        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.15),

        # ── E. Anti jalan-pintas warna ────────────────────────────────────────
        build_decolor_group(strength),
    ], p=1.0)


def build_eval_transform():
    """
    Transform untuk val/test: TIDAK ada augmentasi sama sekali.
    Resize/letterbox ditangani di luar (tf.data), jadi ini identity.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")
    return A.Compose([])


def build_color_stress_transform():
    """
    Test set COLOR-STRESS: warna dibuang total, sisanya dibiarkan apa adanya.

    Ini regression test untuk jalan pintas warna. Model yang benar-benar
    membaca angka nominal harus tetap jauh di atas tebak acak (14.3% untuk
    7 kelas) di sini. Model lama dapat 15.2%, praktis tebak acak.

    Angka ini yang harus kamu pantau setelah retrain, BUKAN akurasi test biasa.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations belum terinstall.")
    return A.Compose([A.ToGray(p=1.0)], p=1.0)


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
