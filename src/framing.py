"""
src/framing.py
==============
Simulasi FRAMING KAMERA: menempel crop uang yang rapat ke dalam "adegan"
buatan (kanvas berbagai rasio + latar + skala + posisi acak).

KENAPA MODUL INI ADA
--------------------
Dataset training kita berisi crop RAPAT hasil bbox YOLO: uang mengisi
>80% bidang, rasionya ~2:1 (landscape). Setelah letterbox 224x224, uang
memenuhi hampir seluruh kanvas.

Di lapangan yang masuk ke model adalah FRAME KAMERA, bukan crop rapat:
uang cuma sebagian kecil bidang, ada tangan, meja, ruangan, dan kalau
frame-nya portrait (9:19.5) letterbox menyisakan bar hitam >50% kanvas.
Contoh nyata: screenshot 720x1560 -> uang menyusut jadi ~95x70 px di
dalam kanvas 224x224. Angka nominal tinggal ~4 px, tidak terbaca.

Model tidak pernah melihat kondisi seperti itu waktu training, jadi dia
ekstrapolasi asal - biasanya ke kelas dengan prior warna terang/besar
(50rb). Ini murni pergeseran distribusi (domain shift) skala + framing,
bukan bug arsitektur.

Modul ini menutup gap itu: waktu training, sebagian sampel sengaja
"dikembalikan" ke bentuk frame kamera penuh.

URUTAN PIPELINE YANG BENAR
--------------------------
    1. decode crop rapat (uang saja)
    2. geometri + keausan  -> dengan MASK, supaya kita tahu di mana uangnya
    3. compose_scene()     -> tempel ke kanvas rasio acak + latar acak   <-- modul ini
    4. fotometri + optik   -> cahaya/blur/noise/JPEG untuk SELURUH adegan
    5. letterbox + normalisasi

Tahap 4 sengaja SETELAH tahap 3 karena secara fisik cahaya, guncangan
tangan, dan kompresi JPEG mengenai seluruh foto, bukan cuma lembar uang.
Kode lama menjalankan semuanya di crop rapat, jadi latar dan uang punya
karakter noise yang berbeda - petunjuk gratis buat model untuk curang.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import IMG_EXTENSIONS

# ─── Distribusi rasio kanvas ───────────────────────────────────────────────────
#
# (lebar, tinggi, bobot). Sengaja berat di portrait karena aplikasi GUIDIO
# dipakai sambil berdiri memegang HP tegak, dan justru itu kasus yang
# sekarang gagal.
CANVAS_ASPECTS: list[tuple[float, float, float]] = [
    (9.0, 19.5, 0.26),   # frame HP portrait modern (kasus yang gagal)
    (9.0, 16.0, 0.20),   # portrait 16:9
    (3.0, 4.0, 0.16),    # portrait 4:3 (rasio kamera default banyak HP)
    (1.0, 1.0, 0.12),    # persegi
    (4.0, 3.0, 0.16),    # landscape 4:3
    (16.0, 9.0, 0.10),   # landscape lebar
]


def sample_canvas_size(rng: np.random.Generator, base: int = 640) -> tuple[int, int]:
    """Pilih (lebar, tinggi) kanvas adegan dari CANVAS_ASPECTS."""
    weights = np.array([a[2] for a in CANVAS_ASPECTS], dtype=np.float64)
    weights /= weights.sum()
    idx = int(rng.choice(len(CANVAS_ASPECTS), p=weights))
    aw, ah, _ = CANVAS_ASPECTS[idx]

    # Normalisasi supaya sisi terpanjang = base, lalu jitter sedikit
    scale = base / max(aw, ah)
    jitter = float(rng.uniform(0.85, 1.15))
    w = max(64, int(round(aw * scale * jitter)))
    h = max(64, int(round(ah * scale * jitter)))
    return w, h


# ─── Latar belakang ────────────────────────────────────────────────────────────

class BackgroundPool:
    """
    Kumpulan gambar latar. Kalau `bg_dir` diberikan, foto-foto di sana
    dipakai (idealnya: foto tangan, meja, lantai, warung, dompet - TANPA
    uang di dalamnya). Kalau tidak ada, jatuh ke latar prosedural.

    Latar prosedural jauh lebih lemah daripada foto asli, tapi tetap jauh
    lebih baik daripada bar hitam: yang penting model belajar "uang itu
    objek DI DALAM adegan", bukan "uang itu seluruh gambar".
    """

    def __init__(self, bg_dir: str | Path | None = None, max_images: int = 400,
                 cache_side: int = 720):
        self.paths: list[Path] = []
        self.cache_side = cache_side
        self._cache: dict[int, np.ndarray] = {}

        if bg_dir:
            d = Path(bg_dir)
            if d.exists():
                self.paths = [
                    p for p in sorted(d.rglob("*"))
                    if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
                ][:max_images]

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, idx: int) -> np.ndarray | None:
        if idx in self._cache:
            return self._cache[idx]
        img = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = self.cache_side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA)
        self._cache[idx] = img
        return img

    def get(self, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
        """Return latar RGB uint8 berukuran (h, w, 3)."""
        if self.paths:
            img = self._load(int(rng.integers(0, len(self.paths))))
            if img is not None:
                return _random_cover_crop(img, w, h, rng)
        return _procedural_background(w, h, rng)


def _random_cover_crop(img: np.ndarray, w: int, h: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Skala latar supaya menutupi (w, h) lalu ambil potongan acak."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih) * float(rng.uniform(1.0, 1.4))
    nw, nh = max(w, int(round(iw * scale))), max(h, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = int(rng.integers(0, nw - w + 1))
    y = int(rng.integers(0, nh - h + 1))
    return np.ascontiguousarray(resized[y:y + h, x:x + w])


def _procedural_background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """
    Latar sintetis: bidang warna + gradien + tekstur noise low-frequency,
    kadang ditambah garis-garis (ubin/meja kayu) atau vignette.

    Palet sengaja condong ke warna yang sering jadi latar uang di Indonesia:
    kulit tangan, kayu meja, lantai keramik abu, aspal, kain gelap.
    """
    palettes = np.array([
        [206, 170, 140],  # kulit terang
        [150, 110,  84],  # kulit gelap / kayu
        [120,  84,  56],  # meja kayu
        [196, 196, 190],  # keramik abu terang
        [ 96,  98, 102],  # aspal / lantai abu
        [ 48,  50,  58],  # ruangan gelap / kain
        [176, 186, 168],  # dinding hijau pudar
        [222, 214, 198],  # kertas / dinding krem
    ], dtype=np.float32)
    base = palettes[int(rng.integers(0, len(palettes)))]
    base = base * float(rng.uniform(0.55, 1.25))

    canvas = np.empty((h, w, 3), dtype=np.float32)
    canvas[:] = base

    # Gradien pencahayaan arah acak
    ang = float(rng.uniform(0, 2 * np.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (np.cos(ang) * xx / max(w, 1) + np.sin(ang) * yy / max(h, 1))
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
    canvas *= (1.0 + (ramp[..., None] - 0.5) * float(rng.uniform(0.15, 0.55)))

    # Tekstur low-frequency (serat kayu / bayangan lembut)
    noise = rng.random((h, w)).astype(np.float32) - 0.5
    sigma = float(rng.uniform(max(w, h) / 40.0, max(w, h) / 8.0))
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    m = np.abs(noise).max()
    if m > 1e-6:
        noise /= m
    canvas *= (1.0 + noise[..., None] * float(rng.uniform(0.08, 0.30)))

    # Kadang: garis nat ubin / papan kayu
    if rng.random() < 0.30:
        vertical = rng.random() < 0.5
        step = int(rng.integers(max(12, min(w, h) // 8), max(20, min(w, h) // 2)))
        dark = float(rng.uniform(0.55, 0.88))
        thick = max(1, int(rng.integers(1, 4)))
        length = w if vertical else h
        for pos in range(int(rng.integers(0, step)), length, step):
            if vertical:
                canvas[:, pos:pos + thick] *= dark
            else:
                canvas[pos:pos + thick, :] *= dark

    # Grain sensor
    canvas += rng.normal(0.0, float(rng.uniform(1.5, 7.0)), size=canvas.shape)

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ─── Compositing ───────────────────────────────────────────────────────────────

def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box piksel non-nol pada mask. Return (x1, y1, x2, y2) eksklusif."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_for_rotation(img: np.ndarray, frac: float = 0.42
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Beri margin nol di sekeliling crop SEBELUM transform geometris,
    dan buat mask yang menandai di mana uang aslinya.

    Ini memperbaiki bug diam-diam di pipeline lama: `A.Affine(rotate=(-180,180))`
    mempertahankan ukuran kanvas, jadi crop 400x200 yang diputar 90 derajat
    terpotong jadi 200x200 - ujung kiri-kanan uang (tempat angka nominal
    berada) hilang. Model dilatih pada uang buntung tanpa ada yang sadar.

    Returns:
        (img_padded uint8 HWC, mask_padded uint8 HW bernilai 0/255)
    """
    h, w = img.shape[:2]
    py = int(round(h * frac))
    px = int(round(w * frac))

    padded = cv2.copyMakeBorder(img, py, py, px, px,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask = np.zeros(padded.shape[:2], dtype=np.uint8)
    mask[py:py + h, px:px + w] = 255
    return padded, mask


def compose_scene(
    note: np.ndarray,
    mask: np.ndarray,
    bg_pool: BackgroundPool,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.28, 1.0),
    base_side: int = 640,
    edge_softness: float = 1.2,
) -> np.ndarray:
    """
    Tempel `note` (dipandu `mask`) ke kanvas adegan berukuran & rasio acak.

    Args:
        note: uint8 HWC RGB, hasil tahap geometri (boleh punya area nol)
        mask: uint8 HW, 255 di piksel uang, 0 di luar
        scale_range: fraksi sisi TERPANJANG kanvas yang ditempati sisi
            terpanjang uang. 1.0 = uang memenuhi kanvas (perilaku lama),
            0.28 = uang kecil di tengah frame (kasus lapangan yang gagal).

            Kenapa relatif ke sisi terpanjang: setelah letterbox, sisi
            terpanjang kanvas dipetakan ke 224 px. Jadi scale=0.4 berarti
            uang jadi ~90 px di tensor 224x224 - persis skala pada fixture
            screenshot yang bikin model salah.

    Returns:
        uint8 HWC RGB adegan lengkap.
    """
    box = _mask_bbox(mask)
    if box is None:
        return note
    x1, y1, x2, y2 = box
    note = note[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    nh, nw = note.shape[:2]
    if nh < 2 or nw < 2:
        return note

    cw, ch = sample_canvas_size(rng, base=base_side)
    canvas_long = max(cw, ch)

    s = float(rng.uniform(*scale_range))
    target_long = max(8, int(round(canvas_long * s)))
    note_long = max(nh, nw)
    f = target_long / note_long

    tw = max(4, int(round(nw * f)))
    th = max(4, int(round(nh * f)))

    # Kalau uang tetap kebesaran untuk kanvas ini, kecilkan sampai muat.
    fit = min(1.0, (cw - 2) / tw, (ch - 2) / th)
    if fit < 1.0:
        tw = max(4, int(tw * fit))
        th = max(4, int(th * fit))

    interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR
    note_r = cv2.resize(note, (tw, th), interpolation=interp)
    mask_r = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_LINEAR)

    bg = bg_pool.get(cw, ch, rng)

    # Posisi acak, condong ke tengah (pengguna diarahkan bingkai panduan
    # di layar, jadi uang biasanya memang di sekitar tengah - tapi tidak
    # pernah persis di tengah).
    max_x = max(0, cw - tw)
    max_y = max(0, ch - th)
    ox = int(np.clip(rng.normal(max_x / 2.0, max(1.0, max_x / 5.0)), 0, max_x))
    oy = int(np.clip(rng.normal(max_y / 2.0, max(1.0, max_y / 5.0)), 0, max_y))

    alpha = mask_r.astype(np.float32) / 255.0
    if edge_softness > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=edge_softness)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    region = bg[oy:oy + th, ox:ox + tw].astype(np.float32)
    blended = note_r.astype(np.float32) * alpha + region * (1.0 - alpha)
    bg[oy:oy + th, ox:ox + tw] = np.clip(blended, 0, 255).astype(np.uint8)

    return bg


class SceneFramer:
    """Pembungkus stateful supaya BackgroundPool di-cache lintas pemanggilan."""

    def __init__(self, bg_dir=None, scale_range=(0.28, 1.0), base_side=640,
                 pad_frac=0.42):
        self.pool = BackgroundPool(bg_dir)
        self.scale_range = scale_range
        self.base_side = base_side
        self.pad_frac = pad_frac

    def prepare(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pad_for_rotation(img, self.pad_frac)

    def compose(self, note: np.ndarray, mask: np.ndarray,
                rng: np.random.Generator,
                scale_range: tuple[float, float] | None = None) -> np.ndarray:
        # scale_range dilewatkan per-panggilan, TIDAK disimpan di self:
        # map tf.data jalan multi-thread, jadi mengubah atribut objek di
        # sini akan saling menimpa antar sampel tanpa error apa pun.
        return compose_scene(note, mask, self.pool, rng,
                             scale_range=scale_range or self.scale_range,
                             base_side=self.base_side)
