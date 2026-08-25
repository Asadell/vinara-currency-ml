"""
src/data.py
===========
Pipeline tf.data untuk rupiah-vision.

Poin desain penting:

  1. LETTERBOX, bukan squash
     `tf.image.resize_with_pad` menjaga aspect ratio dan mengisi sisa
     dengan 0. Setelah normalisasi (x/127.5 - 1), nilai 0 jadi -1.0,
     yang artinya padding = hitam. Konsisten dan tidak bikin artefak aneh.

  2. AUGMENTASI DI RUANG uint8, SEBELUM RESIZE
     Albumentations bekerja di uint8 pada crop ukuran asli. Kalau
     augmentasi dilakukan setelah resize+normalisasi, distorsi geometris
     jadi tidak realistis dan noise/JPEG artifact-nya salah skala.

  3. BALANCED SAMPLING, bukan class_weight
     `class_weight` di Keras tidak jalan dengan label one-hot (yang kita
     butuhkan untuk MixUp/CutMix + label smoothing). Solusinya:
     `tf.data.Dataset.sample_from_datasets` dengan bobot seragam per kelas.
     Efeknya sama (tiap kelas muncul sama sering) tapi kompatibel dengan
     one-hot dan lebih stabil daripada mengalikan loss.

  4. MIXUP & CUTMIX
     Diterapkan di level batch setelah augmentasi per-gambar.
     Untuk klasifikasi nominal, MixUp berguna sebagai regularizer dan
     kalibrator confidence (model jadi tidak terlalu pede), tapi porsinya
     harus kecil (alpha rendah, probabilitas sedang). Kalau kebesaran,
     model malah belajar "campuran" yang tidak pernah ada di dunia nyata.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf

from .common import CLASS_ORDER, IMG_EXTENSIONS, NUM_CLASSES
from .framing import SceneFramer

AUTOTUNE = tf.data.AUTOTUNE


# ─── Pemantau kegagalan augmentasi ─────────────────────────────────────────────
#
# Semua map-fn augmentasi membungkus transform Albumentations dalam try/except
# supaya satu gambar bermasalah tidak menghentikan training berjam-jam. Itu
# benar, TAPI versi lama menelan error tanpa jejak sama sekali.
#
# Bahayanya nyata: kode ini memakai API albumentations yang relatif baru
# (`fill`, `fill_mask`, `num_holes_range`, `std_range`, `scale_range`,
# `interpolation_pair`). Kalau versi yang terinstall sedikit berbeda, SETIAP
# panggilan transform melempar exception, augmentasi mati total, dan training
# tetap berjalan mulus tanpa satu pun pesan. Kamu baru sadar berminggu-minggu
# kemudian saat modelnya rapuh di lapangan dan tidak tahu kenapa.
#
# Sekarang kegagalan dihitung dan dilaporkan.

class _AugFailureMonitor:
    """Hitung kegagalan transform, laporkan sekali-sekali (thread-safe)."""

    def __init__(self, warn_at: int = 32, warn_ratio: float = 0.02):
        self._lock = threading.Lock()
        self.total = 0
        self.failed = 0
        self.warn_at = warn_at
        self.warn_ratio = warn_ratio
        self._next_warn = warn_at
        self.first_error = ""

    def record(self, ok: bool, err: str = "") -> None:
        with self._lock:
            self.total += 1
            if ok:
                return
            self.failed += 1
            if not self.first_error:
                self.first_error = err
            if self.failed < self._next_warn:
                return
            ratio = self.failed / max(1, self.total)
            self._next_warn *= 2
            if ratio < self.warn_ratio:
                return
            print(
                f"\n[AUGMENTASI] {self.failed}/{self.total} sampel "
                f"({ratio * 100:.1f}%) GAGAL diaugmentasi dan dipakai apa "
                f"adanya.\n"
                f"              Error pertama: {self.first_error}\n"
                f"              Kalau rasionya tinggi, augmentasi kamu praktis "
                f"MATI. Cek versi albumentations "
                f"(requirements.txt minta >= 1.4.15).",
                file=sys.stderr, flush=True,
            )

    def summary(self) -> dict:
        with self._lock:
            return {
                "total": self.total,
                "failed": self.failed,
                "ratio": self.failed / max(1, self.total),
                "first_error": self.first_error,
            }


AUG_FAILURES = _AugFailureMonitor()


# ─── Scanning ──────────────────────────────────────────────────────────────────

def scan_split(data_dir: Path, split: str) -> tuple[list[str], list[int]]:
    """Return (paths, label_indices) untuk satu split."""
    split_dir = Path(data_dir) / split
    paths: list[str] = []
    labels: list[int] = []
    for idx, cls in enumerate(CLASS_ORDER):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTENSIONS:
                paths.append(str(p))
                labels.append(idx)
    return paths, labels


def scan_split_per_class(data_dir: Path, split: str) -> list[list[str]]:
    """Return list of path-lists, satu list per kelas (urutan CLASS_ORDER)."""
    split_dir = Path(data_dir) / split
    out: list[list[str]] = []
    for cls in CLASS_ORDER:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            out.append([])
            continue
        out.append([
            str(p) for p in sorted(cls_dir.iterdir())
            if p.suffix.lower() in IMG_EXTENSIONS
        ])
    return out


# ─── Preprocess dasar ──────────────────────────────────────────────────────────

def decode_uint8(path: tf.Tensor) -> tf.Tensor:
    """Baca file -> uint8 HWC RGB, ukuran asli."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def letterbox_and_normalize(img_uint8: tf.Tensor, img_size: int) -> tf.Tensor:
    """
    Letterbox ke img_size x img_size (jaga aspect ratio, pad 0),
    lalu normalisasi ke [-1, 1] sesuai ekspektasi MobileNetV2.
    """
    img = tf.cast(img_uint8, tf.float32)
    img = tf.image.resize_with_pad(
        img, img_size, img_size, method=tf.image.ResizeMethod.BILINEAR
    )
    img = img / 127.5 - 1.0
    return tf.clip_by_value(img, -1.0, 1.0)


def make_preprocess_fn(img_size: int):
    """Fungsi preprocess tanpa augmentasi (untuk val/test/kalibrasi)."""
    def _fn(path, label):
        img = decode_uint8(path)
        img = letterbox_and_normalize(img, img_size)
        return img, label
    return _fn


# ─── Augmentasi via Albumentations ─────────────────────────────────────────────

def make_albumentations_map_fn(transform, img_size: int):
    """
    Bungkus transform Albumentations supaya bisa dipakai di tf.data.

    Catatan performa: `tf.numpy_function` menjalankan kode Python, jadi
    terikat GIL. Untungnya sebagian besar operasi OpenCV melepas GIL,
    jadi dengan num_parallel_calls=AUTOTUNE throughput-nya tetap memadai.
    Kalau CPU jadi bottleneck (GPU idle), turunkan `--aug-strength` atau
    naikkan `--batch-size`.
    """
    def _augment_numpy(img_np):
        # img_np: uint8 HWC RGB ukuran asli
        try:
            out = transform(image=img_np)["image"]
            AUG_FAILURES.record(True)
        except Exception as exc:
            # Kalau satu transform gagal (misal gambar terlalu kecil untuk
            # kernel blur), jangan hentikan training. Pakai gambar asli,
            # TAPI catat supaya kegagalan sistematis tidak lolos diam-diam.
            AUG_FAILURES.record(False, f"{type(exc).__name__}: {exc}")
            out = img_np
        return np.ascontiguousarray(out, dtype=np.uint8)

    def _fn(path, label):
        img = decode_uint8(path)
        aug = tf.numpy_function(_augment_numpy, [img], tf.uint8, stateful=True)
        aug.set_shape([None, None, 3])
        img = letterbox_and_normalize(aug, img_size)
        return img, label

    return _fn


def make_tf_native_augment_fn(img_size: int):
    """
    Fallback augmentasi pakai TF murni kalau albumentations tidak terinstall.
    Lebih lemah (tidak ada elastic/perspective/crumple), tapi tetap jauh
    lebih baik daripada versi lama.
    """
    def _fn(path, label):
        img = decode_uint8(path)
        img = tf.cast(img, tf.float32)

        # Rotasi 180 derajat (uang kebalik) - realistis
        if_flip = tf.random.uniform([]) < 0.5
        img = tf.cond(if_flip, lambda: tf.image.rot90(img, k=2), lambda: img)

        img = tf.image.random_brightness(img, max_delta=45.0)
        img = tf.image.random_contrast(img, lower=0.65, upper=1.35)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        img = tf.image.random_hue(img, max_delta=0.022)  # ~8 derajat
        img = tf.clip_by_value(img, 0.0, 255.0)

        img = tf.image.resize_with_pad(img, img_size, img_size)

        # Random zoom-in via crop
        do_crop = tf.random.uniform([]) < 0.5
        def _crop():
            size = tf.random.uniform([], int(img_size * 0.8), img_size,
                                     dtype=tf.int32)
            c = tf.image.random_crop(img, size=[size, size, 3])
            return tf.image.resize(c, [img_size, img_size])
        img = tf.cond(do_crop, _crop, lambda: img)

        img = img / 127.5 - 1.0
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, label

    return _fn


def make_scene_augment_map_fn(
    geo_transform,
    photo_transform,
    framer: SceneFramer,
    img_size: int,
    frame_prob: float = 0.75,
    tight_scale: tuple[float, float] = (0.90, 1.0),
):
    """
    Map-fn DUA TAHAP dengan simulasi framing kamera.

    Alur per gambar:
        crop rapat
          -> pad_for_rotation()          (margin, supaya rotasi tidak memotong)
          -> geo_transform(image, mask)  (rotasi, perspektif, lecek, sobek)
          -> compose_scene()             (tempel ke kanvas rasio & skala acak)
          -> photo_transform(image)      (cahaya, blur, noise, JPEG)
          -> letterbox + normalisasi

    `frame_prob` = peluang sampel dipakai sebagai ADEGAN PENUH (uang kecil
    di dalam frame). Sisanya dikomposisi rapat (`tight_scale`) supaya model
    tetap tajam di kasus ideal - uang mengisi bingkai panduan di layar.

    Kenapa tidak 100% adegan penuh: aplikasi memang mengarahkan pengguna
    lewat bingkai panduan, jadi crop rapat tetap distribusi yang paling
    sering. Yang kita perbaiki adalah EKORNYA, bukan menggantinya.
    """
    wide_scale = framer.scale_range

    def _augment_numpy(img_np):
        rng = np.random.default_rng()
        try:
            padded, mask = framer.prepare(img_np)
            out = geo_transform(image=padded, mask=mask)
            note, mask = out["image"], out["mask"]

            scale = wide_scale if rng.random() < frame_prob else tight_scale
            scene = framer.compose(note, mask, rng, scale_range=scale)

            scene = photo_transform(image=scene)["image"]
            AUG_FAILURES.record(True)
        except Exception as exc:
            # Satu transform gagal (gambar terlalu kecil untuk kernel blur,
            # mask kosong, dll) tidak boleh menghentikan training. Tapi kalau
            # SEMUA gagal, simulasi framing praktis mati dan kamu harus tahu.
            AUG_FAILURES.record(False, f"{type(exc).__name__}: {exc}")
            scene = img_np
        return np.ascontiguousarray(scene, dtype=np.uint8)

    def _fn(path, label):
        img = decode_uint8(path)
        aug = tf.numpy_function(_augment_numpy, [img], tf.uint8, stateful=True)
        aug.set_shape([None, None, 3])
        img = letterbox_and_normalize(aug, img_size)
        return img, label

    return _fn


def make_scene_eval_map_fn(framer: SceneFramer, img_size: int,
                           scale_range: tuple[float, float] = (0.30, 0.45)):
    """
    Map-fn evaluasi untuk "FRAMING EVAL": uang bersih ditempel ke frame
    portrait pada skala kecil, TANPA degradasi lain.

    Gunanya regression test terarah: kalau akurasi di sini jauh di bawah
    test biasa, artinya model masih rapuh terhadap framing - persis
    kegagalan pada fixture screenshot 720x1560.
    """
    def _compose_numpy(img_np):
        rng = np.random.default_rng(abs(int(img_np.sum())) % (2 ** 31))
        try:
            padded, mask = framer.prepare(img_np)
            scene = framer.compose(padded, mask, rng, scale_range=scale_range)
        except Exception:
            scene = img_np
        return np.ascontiguousarray(scene, dtype=np.uint8)

    def _fn(path, label):
        img = decode_uint8(path)
        aug = tf.numpy_function(_compose_numpy, [img], tf.uint8, stateful=True)
        aug.set_shape([None, None, 3])
        img = letterbox_and_normalize(aug, img_size)
        return img, label

    return _fn


# ─── MixUp & CutMix ────────────────────────────────────────────────────────────

def _sample_beta(alpha: float, shape) -> tf.Tensor:
    """Sampel dari distribusi Beta(alpha, alpha) pakai dua Gamma."""
    g1 = tf.random.gamma(shape, alpha)
    g2 = tf.random.gamma(shape, alpha)
    return g1 / (g1 + g2 + 1e-9)


def apply_mixup(images, labels, alpha: float):
    """MixUp: campur dua gambar dan dua label secara linear."""
    batch = tf.shape(images)[0]
    lam = _sample_beta(alpha, [batch])
    # Pastikan lam >= 0.5 supaya gambar utama tetap dominan
    lam = tf.maximum(lam, 1.0 - lam)

    idx = tf.random.shuffle(tf.range(batch))
    img_b = tf.gather(images, idx)
    lbl_b = tf.gather(labels, idx)

    lam_img = tf.reshape(lam, [batch, 1, 1, 1])
    lam_lbl = tf.reshape(lam, [batch, 1])

    mixed_img = images * lam_img + img_b * (1.0 - lam_img)
    mixed_lbl = labels * lam_lbl + lbl_b * (1.0 - lam_lbl)
    return mixed_img, mixed_lbl


def apply_cutmix(images, labels, alpha: float):
    """CutMix: tempel potongan persegi dari gambar lain."""
    batch = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    lam = _sample_beta(alpha, [])
    lam = tf.maximum(lam, 1.0 - lam)

    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * cut_ratio, tf.int32)

    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)

    idx = tf.random.shuffle(tf.range(batch))
    img_b = tf.gather(images, idx)
    lbl_b = tf.gather(labels, idx)

    # Bangun mask persegi
    yy = tf.range(h)[:, None]
    xx = tf.range(w)[None, :]
    mask = tf.cast(
        (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2), tf.float32
    )[None, :, :, None]

    mixed_img = images * (1.0 - mask) + img_b * mask

    area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    total = tf.cast(h * w, tf.float32)
    real_lam = 1.0 - area / total
    mixed_lbl = labels * real_lam + lbl_b * (1.0 - real_lam)
    return mixed_img, mixed_lbl


def make_batch_mix_fn(mixup_alpha: float, cutmix_alpha: float,
                      mix_prob: float):
    """
    Return fungsi batch-level yang menerapkan MixUp ATAU CutMix
    dengan probabilitas mix_prob (dipilih 50:50 di antara keduanya).
    """
    def _fn(images, labels):
        r = tf.random.uniform([])

        def _no_mix():
            return images, labels

        def _do_mix():
            r2 = tf.random.uniform([])
            return tf.cond(
                r2 < 0.5,
                lambda: apply_mixup(images, labels, mixup_alpha),
                lambda: apply_cutmix(images, labels, cutmix_alpha),
            )

        return tf.cond(r < mix_prob, _do_mix, _no_mix)

    return _fn


# ─── Builder dataset ───────────────────────────────────────────────────────────

def build_train_dataset(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    transform=None,
    geo_transform=None,
    photo_transform=None,
    framer: SceneFramer | None = None,
    frame_prob: float = 0.75,
    balanced: bool = True,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.35,
    shuffle_buffer: int = 4096,
    seed: int = 42,
) -> tuple[tf.data.Dataset, int, list[int]]:
    """
    Bangun dataset training.

    Returns:
        (dataset, steps_per_epoch, class_counts)
    """
    per_class = scan_split_per_class(data_dir, "train")
    class_counts = [len(p) for p in per_class]
    total = sum(class_counts)
    if total == 0:
        raise RuntimeError("Train set kosong.")

    if framer is not None and geo_transform is not None \
            and photo_transform is not None:
        # Jalur baru: dua tahap + simulasi framing kamera.
        map_fn = make_scene_augment_map_fn(
            geo_transform, photo_transform, framer, img_size,
            frame_prob=frame_prob,
        )
    elif transform is not None:
        map_fn = make_albumentations_map_fn(transform, img_size)
    else:
        map_fn = make_tf_native_augment_fn(img_size)

    def _one_hot(img, label):
        return img, tf.one_hot(label, NUM_CLASSES)

    if balanced:
        datasets = []
        weights = []
        for idx, paths in enumerate(per_class):
            if not paths:
                continue
            labels = [idx] * len(paths)
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            ds = ds.shuffle(min(len(paths), shuffle_buffer), seed=seed + idx,
                            reshuffle_each_iteration=True)
            ds = ds.repeat()
            datasets.append(ds)
            weights.append(1.0)

        weights = [w / len(weights) for w in weights]
        ds = tf.data.Dataset.sample_from_datasets(
            datasets, weights=weights, seed=seed, stop_on_empty_dataset=False
        )
        # Satu epoch = seolah-olah tiap kelas muncul sebanyak rata-rata kelas
        n_effective = int(np.mean([c for c in class_counts if c > 0])) * \
            sum(1 for c in class_counts if c > 0)
        steps = max(1, n_effective // batch_size)
    else:
        paths, labels = scan_split(data_dir, "train")
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(min(len(paths), shuffle_buffer), seed=seed,
                        reshuffle_each_iteration=True).repeat()
        steps = max(1, total // batch_size)

    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_one_hot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if mix_prob > 0:
        ds = ds.map(make_batch_mix_fn(mixup_alpha, cutmix_alpha, mix_prob),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(AUTOTUNE)
    return ds, steps, class_counts


def build_eval_dataset(
    data_dir: Path,
    split: str,
    img_size: int,
    batch_size: int,
    transform=None,
    framer: SceneFramer | None = None,
    frame_scale: tuple[float, float] = (0.30, 0.45),
    one_hot: bool = True,
) -> tuple[tf.data.Dataset, int]:
    """
    Bangun dataset val/test tanpa augmentasi (atau dengan transform khusus
    untuk membuat hard-eval set sintetis).

    Returns:
        (dataset, n_images)
    """
    paths, labels = scan_split(data_dir, split)
    if not paths:
        raise RuntimeError(f"Split '{split}' kosong.")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if framer is not None:
        ds = ds.map(make_scene_eval_map_fn(framer, img_size, frame_scale),
                    num_parallel_calls=AUTOTUNE)
    elif transform is not None:
        ds = ds.map(make_albumentations_map_fn(transform, img_size),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(make_preprocess_fn(img_size), num_parallel_calls=AUTOTUNE)

    if one_hot:
        ds = ds.map(lambda i, l: (i, tf.one_hot(l, NUM_CLASSES)),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, len(paths)
