#!/usr/bin/env python3
"""
01_train.py  (REVISI)
=====================
Training MobileNetV2 untuk klasifikasi uang Rupiah, dioptimalkan untuk
ROBUSTNESS di dunia nyata (uang lecek, terlipat, dicoret, cahaya buruk),
bukan cuma akurasi tinggi di dataset bersih.

RINGKASAN PERUBAHAN DARI VERSI LAMA:

  Data & augmentasi
    - Letterbox (jaga aspect ratio) menggantikan squash ke 224x224
    - Augmentasi berat khusus uang lecek: elastic, grid distortion,
      perspective, simulasi lipatan, simulasi kusut, coretan, stempel,
      bayangan, gamma, motion blur, JPEG artifact, downscale
    - Hue shift TERBATAS (+/-8 derajat) supaya model tidak cuma hafal warna
    - Hilangkan horizontal flip (uang cermin tidak pernah ada di dunia nyata),
      diganti rotasi bebas 0-360 yang memang realistis

  Label & loss
    - Label smoothing 0.05 (kurangi overconfidence)
    - MixUp + CutMix probabilistik (regularizer + kalibrator confidence)
    - Balanced sampling menggantikan class_weight (kompatibel dengan one-hot)

  Optimisasi
    - AdamW + weight decay
    - Warmup linear lalu cosine decay
    - EMA (exponential moving average) bobot
    - Tahap 2 unfreeze SEMUA layer dengan LR kecil, BatchNorm dibekukan
      (opsional) supaya statistik ImageNet tidak rusak oleh batch kecil

  Evaluasi
    - Metrik per kelas (precision/recall/F1) + confusion matrix
    - HARD EVAL: test set yang sengaja didegradasi (lecek + gelap + blur)
      supaya kamu tahu kondisi model di skenario terburuk, bukan cuma
      angka cantik di uang bersih
    - Kurva reliability + Expected Calibration Error (ECE)

Usage:
    python scripts/01_train.py \
        --data data/classification \
        --output models \
        --aug-strength medium \
        --head-epochs 12 \
        --finetune-epochs 45 \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import CLASS_ORDER, CLASS_TO_SPOKEN, NUM_CLASSES  # noqa: E402
from src.data import (  # noqa: E402
    AUG_FAILURES,
    build_eval_dataset,
    build_train_dataset,
)

from src.framing import SceneFramer  # noqa: E402

try:
    from src.augment import (
        build_color_stress_transform,
        build_geometric_transform,
        build_hard_eval_transform,
        build_photometric_transform,
        build_train_transform,
    )
    HAS_ALBU = True
except ImportError:
    HAS_ALBU = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Learning rate schedule
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup linear dari 0 ke peak_lr selama warmup_steps,
    lalu cosine decay ke min_lr sampai total_steps.

    Warmup penting waktu fine-tuning semua layer: tanpa warmup,
    beberapa step pertama dengan gradien besar bisa merusak fitur
    pretrained ImageNet sebelum head-nya sempat menyesuaikan.
    """

    def __init__(self, peak_lr: float, total_steps: int,
                 warmup_steps: int, min_lr: float = 1e-7):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.peak_lr * (step / warmup)

        progress = tf.clip_by_value(
            (step - warmup) / tf.maximum(total - warmup, 1.0), 0.0, 1.0
        )
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * \
            (1.0 + tf.cos(math.pi * progress))

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(img_size: int, dropout: float = 0.35,
                width_alpha: float = 1.0,
                weights: str | None = "imagenet") -> tuple[keras.Model, keras.Model]:
    """
    MobileNetV2 + head klasifikasi.

    Head sengaja dibuat lebih ramping dari versi lama (Dense 256 -> 128 -> 7).
    Head tebal di atas backbone beku gampang overfit ke dataset kecil.
    Satu Dense 128 + dropout sudah cukup, dan sisa kapasitasnya lebih baik
    dialokasikan ke fine-tuning backbone.

    Output sengaja LOGITS (tanpa softmax) supaya:
      - loss lebih stabil secara numerik (from_logits=True)
      - kalibrasi temperature scaling di 03_calibrate_threshold.py bisa
        bekerja langsung di logits
    Softmax ditempel belakangan waktu ekspor untuk deployment.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights,
        alpha=width_alpha,
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.Dense(128, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout * 0.6, name="drop2")(x)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)

    model = keras.Model(inputs, outputs, name="rupiah_mobilenetv2")
    return model, base


def set_finetune_mode(base: keras.Model, freeze_bn: bool,
                      freeze_first_n: int = 0) -> int:
    """
    Buka semua layer backbone untuk fine-tuning.

    freeze_bn=True membekukan seluruh BatchNormalization. Alasannya:
    dengan batch kecil (16-32) dan augmentasi berat, statistik batch jadi
    berisik. Membiarkan BN meng-update running stats sering justru
    menurunkan akurasi validasi di awal fine-tuning. Kalau batch kamu
    besar (>= 64) dan datanya banyak, coba --no-freeze-bn.

    freeze_first_n membekukan N layer paling awal (fitur tepi/tekstur
    generik yang tidak perlu diubah). Default 0 = buka semua.
    """
    base.trainable = True
    n_trainable = 0
    for i, layer in enumerate(base.layers):
        if i < freeze_first_n:
            layer.trainable = False
            continue
        if freeze_bn and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = True
        n_trainable += 1
    return n_trainable


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluasi
# ═══════════════════════════════════════════════════════════════════════════════

def predict_logits(model: keras.Model, ds: tf.data.Dataset):
    """Kumpulkan logits + label sebenarnya dari satu dataset."""
    all_logits, all_true = [], []
    for images, labels in ds:
        logits = model(images, training=False)
        all_logits.append(np.asarray(logits))
        all_true.append(np.asarray(labels))
    return np.concatenate(all_logits, 0), np.concatenate(all_true, 0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hitung precision/recall/F1 per kelas + confusion matrix."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    report = {}
    for i, cls in enumerate(CLASS_ORDER):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[cls] = {
            "support": int(cm[i, :].sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return {"per_class": report, "confusion_matrix": cm.tolist()}


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray,
                               n_bins: int = 15) -> float:
    """
    ECE: seberapa jauh confidence model dari akurasi sebenarnya.
    Model yang bilang "95% yakin" idealnya benar 95% dari waktu.
    ECE tinggi = model overconfident = bahaya untuk aplikasi uang.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def print_eval(name: str, logits: np.ndarray, y_true_onehot: np.ndarray) -> dict:
    y_true = y_true_onehot.argmax(axis=1)
    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    ece = expected_calibration_error(probs, y_true)
    rep = per_class_report(y_true, y_pred)

    print(f"\n  {name}")
    print(f"  {'-' * 58}")
    print(f"  Akurasi : {acc * 100:.2f}%   ({int((y_pred == y_true).sum())}/{len(y_true)})")
    print(f"  ECE     : {ece:.4f}  "
          f"({'bagus' if ece < 0.05 else 'overconfident, perlu kalibrasi'})")
    print(f"\n  {'Kelas':>8} {'support':>8} {'prec':>7} {'recall':>7} {'F1':>7}")
    for cls in CLASS_ORDER:
        r = rep["per_class"][cls]
        print(f"  {cls:>8} {r['support']:>8} {r['precision']:>7.3f} "
              f"{r['recall']:>7.3f} {r['f1']:>7.3f}")

    print(f"\n  Confusion matrix (baris=asli, kolom=prediksi):")
    header = "  " + " " * 9 + "".join(f"{c:>8}" for c in CLASS_ORDER)
    print(header)
    cm = np.array(rep["confusion_matrix"])
    for i, cls in enumerate(CLASS_ORDER):
        row = "".join(
            (f"{v:>8}" if i != j or v == 0 else f"{v:>7}*")
            for j, v in enumerate(cm[i])
        )
        print(f"  {cls:>8} {row}")

    # Pasangan yang paling sering ketuker
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], CLASS_ORDER[i], CLASS_ORDER[j]))
    confusions.sort(reverse=True)
    if confusions:
        print("\n  Kekeliruan terbanyak:")
        for count, a, b in confusions[:5]:
            print(f"    {a:>7} dibaca sebagai {b:<7} : {count}x")

    rep["accuracy"] = acc
    rep["ece"] = ece
    return rep


def plot_reliability(probs: np.ndarray, y_true: np.ndarray,
                     out_path: Path, n_bins: int = 15) -> None:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    centers, accs, confs = [], [], []
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() < 5:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        accs.append(correct[mask].mean())
        confs.append(conf[mask].mean())

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "k--", label="Kalibrasi sempurna")
    plt.plot(confs, accs, "o-", label="Model")
    plt.xlabel("Confidence rata-rata")
    plt.ylabel("Akurasi sebenarnya")
    plt.title("Reliability diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_history(histories: list, boundaries: list[int], out_path: Path) -> None:
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history.get("acc", h.history.get("accuracy", []))
        val_acc += h.history.get("val_acc", h.history.get("val_accuracy", []))
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(acc, label="Train")
    axes[0].plot(val_acc, label="Val")
    axes[0].set_title("Accuracy")
    axes[1].plot(loss, label="Train")
    axes[1].plot(val_loss, label="Val")
    axes[1].set_title("Loss")
    for ax in axes:
        for b in boundaries:
            ax.axvline(x=b, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Training MobileNetV2 Rupiah (robust ke uang lecek)"
    )
    ap.add_argument("--data", default="data/classification")
    ap.add_argument("--output", default="models")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--head-epochs", type=int, default=12,
                    help="Epoch tahap 1 (backbone beku)")
    ap.add_argument("--finetune-epochs", type=int, default=45,
                    help="Epoch tahap 2 (fine-tune seluruh backbone)")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=4e-5,
                    help="Peak LR tahap 2 (setelah warmup)")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.08,
                    help="Porsi step awal untuk warmup linear")

    # ── Simulasi framing kamera ──────────────────────────────────────────
    ap.add_argument("--frame-prob", type=float, default=0.75,
                    help="Peluang satu sampel training dikomposisi jadi ADEGAN "
                         "PENUH (uang kecil di dalam frame kamera) bukan crop "
                         "rapat. 0 = matikan simulasi framing, kembali ke "
                         "pipeline satu tahap versi lama.")
    ap.add_argument("--frame-scale-min", type=float, default=0.28,
                    help="Fraksi minimum sisi terpanjang kanvas yang ditempati "
                         "uang. 0.28 -> uang jadi ~63px di tensor 224x224.")
    ap.add_argument("--frame-scale-max", type=float, default=1.0)
    ap.add_argument("--bg-dir", default=None,
                    help="Folder foto LATAR (tangan, meja, lantai, warung - "
                         "tanpa uang). Sangat dianjurkan. Kalau kosong, "
                         "dipakai latar prosedural yang jauh lebih lemah.")
    ap.add_argument("--frame-eval-scale", type=float, default=0.38,
                    help="Skala uang untuk FRAMING EVAL (regression test "
                         "khusus kegagalan framing).")
    ap.add_argument("--aug-strength", choices=["light", "medium", "heavy"],
                    default="medium")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--cutmix-alpha", type=float, default=1.0)
    ap.add_argument("--mix-prob", type=float, default=0.35,
                    help="Probabilitas satu batch kena MixUp/CutMix; 0 = matikan")

    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--width-alpha", type=float, default=1.0,
                    help="Width multiplier MobileNetV2 (0.75 kalau butuh lebih ringan)")
    ap.add_argument("--weights", default="imagenet",
                    help="Bobot awal backbone: 'imagenet', path .h5, atau "
                         "'none' untuk latih dari nol (tidak disarankan)")
    ap.add_argument("--no-freeze-bn", action="store_true",
                    help="Biarkan BatchNorm ikut belajar saat fine-tuning")
    ap.add_argument("--freeze-first-n", type=int, default=0,
                    help="Bekukan N layer backbone paling awal")
    ap.add_argument("--no-balanced", action="store_true",
                    help="Matikan balanced sampling per kelas")
    ap.add_argument("--ema", action="store_true",
                    help="Nyalakan exponential moving average bobot. MATI secara "
                         "default karena di pipeline lama dia tidak pernah "
                         "benar-benar terpakai: ModelCheckpoint menyimpan bobot "
                         "MENTAH, dan EarlyStopping(restore_best_weights=True) "
                         "menimpa bobot EMA di akhir fit(). Sudah diverifikasi "
                         "di TF 2.21. Kalau flag ini dinyalakan, "
                         "restore_best_weights otomatis dimatikan supaya bobot "
                         "EMA benar-benar jadi model akhir.")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n==============================================================")
    print(f"  TENSORFLOW VERSION: {tf.__version__}")
    print(f"  GPU ACCELERATION   : {'✅ TERDETEKSI (' + str(len(gpus)) + ' GPU)' if gpus else '❌ GPU TIDAK TERDETEKSI (Menggunakan CPU)'}")
    for gpu in gpus:
        print(f"  🚀 GPU DEVICE      : {gpu.name}")
    print(f"==============================================================\n")
    if gpus:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("  ⚡ Mixed Precision FP16 DIAKTIFKAN untuk GPU Acceleration maksimal!")
        except Exception as e:
            print(f"  ⚠️ Mixed precision warning: {e}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    if not HAS_ALBU:
        print("\nPERINGATAN: albumentations tidak terinstall. "
              "Pipeline jatuh ke augmentasi TF bawaan yang jauh lebih lemah "
              "(tanpa elastic/perspective/simulasi lipatan). "
              "Jalankan: pip install albumentations")

    # ── Dataset ──
    print("\nMenyiapkan dataset...")
    train_transform = build_train_transform(args.aug_strength) if HAS_ALBU else None

    # Simulasi framing kamera: dataset kita cuma berisi crop rapat bbox YOLO
    # (uang >80% bidang), sedangkan yang masuk ke model di lapangan adalah
    # frame kamera penuh dengan uang cuma sebagian kecil bidang. Tanpa ini
    # model tidak pernah melihat skala yang sebenarnya dia hadapi.
    framer = geo_transform = photo_transform = None
    if HAS_ALBU and args.frame_prob > 0:
        framer = SceneFramer(
            bg_dir=args.bg_dir,
            scale_range=(args.frame_scale_min, args.frame_scale_max),
        )
        geo_transform = build_geometric_transform(args.aug_strength)
        photo_transform = build_photometric_transform(args.aug_strength)
        src = f"{len(framer.pool)} foto dari {args.bg_dir}" if len(framer.pool) \
            else "PROSEDURAL (tidak ada --bg-dir)"
        print(f"  Simulasi framing: AKTIF | p={args.frame_prob} | "
              f"skala={args.frame_scale_min}-{args.frame_scale_max} | "
              f"latar={src}")
        if not len(framer.pool):
            print("  SARAN: isi --bg-dir dengan ~200-400 foto tangan/meja/"
                  "lantai/warung tanpa uang. Latar asli jauh lebih efektif "
                  "daripada latar sintetis.")
    else:
        print("  Simulasi framing: MATI (pipeline satu tahap versi lama)")

    train_ds, steps_per_epoch, class_counts = build_train_dataset(
        data_dir, args.img_size, args.batch_size,
        transform=train_transform,
        geo_transform=geo_transform,
        photo_transform=photo_transform,
        framer=framer,
        frame_prob=args.frame_prob,
        balanced=not args.no_balanced,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        seed=args.seed,
    )
    val_ds, n_val = build_eval_dataset(data_dir, "val", args.img_size,
                                       args.batch_size)
    test_ds, n_test = build_eval_dataset(data_dir, "test", args.img_size,
                                         args.batch_size)

    # FRAMING EVAL: test set yang sama, tapi tiap uang ditempel ke frame
    # portrait pada skala kecil TANPA degradasi lain. Ini regression test
    # terarah untuk kegagalan yang kita perbaiki. Kalau akurasinya jauh di
    # bawah test biasa, model masih rapuh terhadap framing.
    frame_eval_ds = None
    if framer is not None:
        frame_eval_ds, _ = build_eval_dataset(
            data_dir, "test", args.img_size, args.batch_size,
            framer=framer,
            frame_scale=(args.frame_eval_scale * 0.8,
                         args.frame_eval_scale * 1.2),
        )

    print(f"  Train: {sum(class_counts)} gambar, "
          f"{steps_per_epoch} step/epoch (balanced="
          f"{not args.no_balanced})")
    print(f"  Val  : {n_val} gambar")
    print(f"  Test : {n_test} gambar")
    print(f"  Distribusi train: "
          f"{dict(zip(CLASS_ORDER, class_counts))}")

    # ── Model ──
    print("\nMembangun model...")
    weights = None if str(args.weights).lower() in ("none", "null", "") else args.weights
    if weights is None:
        print("  PERINGATAN: backbone dilatih dari nol tanpa bobot ImageNet. "
              "Dengan dataset seukuran ini hasilnya akan jauh lebih buruk.")
    model, base = build_model(args.img_size, args.dropout, args.width_alpha, weights)
    n_params = model.count_params()
    print(f"  Total parameter: {n_params:,}")

    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=args.label_smoothing
    )
    metrics = [keras.metrics.CategoricalAccuracy(name="acc")]

    histories = []
    boundaries = []

    # ══════════════════════════════════════════════════════════════════════════
    #  TAHAP 1: latih head saja
    # ══════════════════════════════════════════════════════════════════════════
    if args.head_epochs > 0:
        print("\n" + "=" * 62)
        print(f"  TAHAP 1: latih head (backbone beku), {args.head_epochs} epoch")
        print("=" * 62)

        total_steps_1 = steps_per_epoch * args.head_epochs
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=WarmupCosine(
                    args.head_lr, total_steps_1,
                    int(total_steps_1 * args.warmup_frac)
                ),
                weight_decay=args.weight_decay,
            ),
            loss=loss_fn,
            metrics=metrics,
        )

        h1 = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            epochs=args.head_epochs,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    str(out_dir / "best_stage1.keras"),
                    monitor="val_acc", mode="max",
                    save_best_only=True, verbose=1,
                ),
                keras.callbacks.CSVLogger(str(out_dir / "history_stage1.csv")),
            ],
            verbose=1,
        )
        histories.append(h1)
        boundaries.append(args.head_epochs)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAHAP 2: fine-tune seluruh backbone
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print(f"  TAHAP 2: fine-tune backbone, {args.finetune_epochs} epoch")
    print("=" * 62)

    freeze_bn = not args.no_freeze_bn
    n_trainable = set_finetune_mode(base, freeze_bn, args.freeze_first_n)
    print(f"  Layer backbone yang dilatih: {n_trainable}/{len(base.layers)}")
    print(f"  BatchNorm dibekukan        : {freeze_bn}")

    total_steps_2 = steps_per_epoch * args.finetune_epochs
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=WarmupCosine(
                args.finetune_lr, total_steps_2,
                int(total_steps_2 * args.warmup_frac)
            ),
            weight_decay=args.weight_decay,
            use_ema=args.ema,
            ema_momentum=0.999,
        ),
        loss=loss_fn,
        metrics=metrics,
    )

    h2 = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(out_dir / "best_stage2.keras"),
                monitor="val_acc", mode="max",
                save_best_only=True, verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_acc", mode="max",
                patience=args.patience,
                # Dengan EMA, bobot akhir adalah rata-ratanya. Mengembalikan
                # checkpoint terbaik justru MEMBUANG hasil EMA (terverifikasi).
                restore_best_weights=not args.ema, verbose=1,
            ),
            keras.callbacks.CSVLogger(str(out_dir / "history_stage2.csv")),
        ],
        verbose=1,
    )
    histories.append(h2)

    # ── Muat bobot terbaik ──
    if args.ema:
        # Model in-memory sudah memegang bobot EMA (finalize_variable_values
        # dipanggil di akhir fit, dan restore_best_weights sengaja dimatikan).
        print("\nMemakai bobot EMA hasil akhir fine-tuning (bukan checkpoint).")
        model.save(str(out_dir / "best_stage2_ema.keras"))
    else:
        best_path = out_dir / "best_stage2.keras"
        if not best_path.exists():
            best_path = out_dir / "best_stage1.keras"
        print(f"\nMemuat model terbaik: {best_path}")
        model = keras.models.load_model(str(best_path), compile=False)

    # ══════════════════════════════════════════════════════════════════════════
    #  EVALUASI
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("  EVALUASI")
    print("=" * 62)

    results = {}

    val_logits, val_true = predict_logits(model, val_ds)
    results["val"] = print_eval("VAL (uang apa adanya)", val_logits, val_true)

    test_logits, test_true = predict_logits(model, test_ds)
    results["test"] = print_eval("TEST (uang apa adanya)", test_logits, test_true)

    # Hard eval: test set didegradasi sengaja
    if HAS_ALBU:
        print("\n  Menyiapkan HARD TEST (uang disimulasikan lecek + gelap + blur)...")
        hard_ds, _ = build_eval_dataset(
            data_dir, "test", args.img_size, args.batch_size,
            transform=build_hard_eval_transform(),
        )
        hard_logits, hard_true = predict_logits(model, hard_ds)
        results["test_hard"] = print_eval(
            "HARD TEST (lecek + gelap + blur sintetis)", hard_logits, hard_true
        )

        gap = results["test"]["accuracy"] - results["test_hard"]["accuracy"]
        print(f"\n  Selisih akurasi bersih vs lecek: {gap * 100:.2f} poin")
        if gap > 0.15:
            print("  Model masih rapuh. Coba --aug-strength heavy, "
                  "atau tambah foto uang lecek asli ke train set.")
        elif gap > 0.07:
            print("  Cukup baik, tapi masih ada ruang perbaikan.")
        else:
            print("  Bagus, model relatif stabil di kondisi buruk.")

    # COLOR-STRESS EVAL: warna dibuang total.
    #
    # INI METRIK PALING PENTING DI SELURUH SCRIPT untuk kasus 20rb vs 50rb.
    #
    # Model pipeline lama diukur: 80.5% normal, 15.2% grayscale, sementara
    # tebak acak 7 kelas = 14.3%. Artinya dia tidak pernah membaca angka
    # nominal sama sekali, cuma mencocokkan histogram warna. Akurasi TEST biasa
    # tidak akan pernah menangkap ini karena di test set warnanya juga utuh.
    if HAS_ALBU:
        print("\n  Menyiapkan COLOR-STRESS TEST (warna dibuang total)...")
        gray_ds, _ = build_eval_dataset(
            data_dir, "test", args.img_size, args.batch_size,
            transform=build_color_stress_transform(),
        )
        gray_logits, gray_true = predict_logits(model, gray_ds)
        results["test_grayscale"] = print_eval(
            "COLOR-STRESS TEST (grayscale, warna dibuang)",
            gray_logits, gray_true
        )

        chance = 1.0 / NUM_CLASSES
        gacc = results["test_grayscale"]["accuracy"]
        print(f"\n  Tebak acak {NUM_CLASSES} kelas   : {chance * 100:.1f}%")
        print(f"  Akurasi tanpa warna    : {gacc * 100:.2f}%")
        if gacc < chance * 1.6:
            print("  GAGAL. Model masih classifier warna murni: buang warnanya "
                  "dan dia menebak acak. Dia belum membaca angka nominal, jadi "
                  "20rb vs 50rb akan tetap ketuker di cahaya redup.")
            print("  Naikkan `decolor` di src/augment.py _PRESETS, atau pakai "
                  "--aug-strength heavy.")
        elif gacc < 0.55:
            print("  Membaik, tapi warna masih jadi tumpuan utama.")
        else:
            print("  Bagus. Model punya isyarat non-warna yang nyata "
                  "(angka/pola/potret), bukan cuma histogram warna.")

    # FRAMING EVAL: uang bersih di dalam frame kamera pada skala kecil.
    # Ini metrik yang paling relevan dengan kegagalan lapangan - akurasi
    # TEST biasa tidak akan pernah menangkapnya karena test set pun berisi
    # crop rapat, sama seperti train set.
    if frame_eval_ds is not None:
        print("\n  Menyiapkan FRAMING TEST (uang kecil di dalam frame "
              f"kamera, skala ~{args.frame_eval_scale})...")
        fr_logits, fr_true = predict_logits(model, frame_eval_ds)
        results["test_framing"] = print_eval(
            "FRAMING TEST (uang kecil di frame kamera)", fr_logits, fr_true
        )

        fgap = results["test"]["accuracy"] - results["test_framing"]["accuracy"]
        print(f"\n  Selisih akurasi crop rapat vs frame penuh: "
              f"{fgap * 100:.2f} poin")
        if fgap > 0.20:
            print("  Model masih bias ke crop rapat. Naikkan --frame-prob, "
                  "turunkan --frame-scale-min, dan isi --bg-dir dengan foto "
                  "latar asli.")
        elif fgap > 0.10:
            print("  Membaik, tapi framing masih jadi titik lemah.")
        else:
            print("  Bagus, model tahan terhadap variasi framing & skala.")
        print("  CATATAN: kalau selisih ini kecil TAPI aplikasi masih salah "
              "di lapangan, masalahnya ada di ROI crop sisi aplikasi, bukan "
              "di model.")

    # Reliability diagram
    plot_reliability(
        softmax_np(test_logits), test_true.argmax(axis=1),
        out_dir / "reliability_test.png"
    )
    plot_history(histories, boundaries, out_dir / "training_history.png")

    # ── Simpan artefak ──
    # Model inferensi = backbone + softmax, supaya siap diekspor
    softmax_out = layers.Activation("softmax", name="probs")(model.output)
    infer_model = keras.Model(model.input, softmax_out, name="rupiah_infer")

    keras_path = out_dir / "rupiah_final.keras"
    model.save(str(keras_path))  # versi logits (untuk kalibrasi)
    infer_path = out_dir / "rupiah_infer.keras"
    infer_model.save(str(infer_path))  # versi softmax (untuk ekspor)

    saved_model_dir = out_dir / "rupiah_mobilenetv2"
    infer_model.export(str(saved_model_dir))

    np.savez(
        out_dir / "test_logits.npz",
        logits=test_logits, labels=test_true.argmax(axis=1),
        val_logits=val_logits, val_labels=val_true.argmax(axis=1),
    )

    class_info = {
        "classes": CLASS_ORDER,
        "spoken": CLASS_TO_SPOKEN,
        "idx_to_class": {str(i): c for i, c in enumerate(CLASS_ORDER)},
        "img_size": args.img_size,
        "preprocessing": {
            "resize": "letterbox (resize_with_pad), aspect ratio dipertahankan",
            "normalization": "x / 127.5 - 1.0  -> rentang [-1, 1]",
            "channel_order": "RGB",
            "pad_value_after_norm": -1.0,
        },
        "training_args": vars(args),
        "n_params": int(n_params),
    }
    with open(out_dir / "class_info.json", "w", encoding="utf-8") as f:
        json.dump(class_info, f, indent=2)
    # Laporan kesehatan augmentasi. Kalau rasionya tinggi, seluruh augmentasi
    # praktis tidak jalan dan semua angka di atas menyesatkan.
    aug_stats = AUG_FAILURES.summary()
    results["augmentation_health"] = aug_stats
    if aug_stats["total"]:
        print(f"\n  Kesehatan augmentasi: {aug_stats['failed']}/"
              f"{aug_stats['total']} gagal "
              f"({aug_stats['ratio'] * 100:.2f}%)")
        if aug_stats["ratio"] > 0.02:
            print(f"  PERINGATAN: augmentasi banyak yang gagal dan gambar "
                  f"dipakai apa adanya.\n"
                  f"  Error pertama: {aug_stats['first_error']}\n"
                  f"  Semua angka evaluasi di atas TIDAK bisa dipercaya "
                  f"sampai ini beres.")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    (out_dir / "labels.txt").write_text("\n".join(CLASS_ORDER) + "\n",
                                        encoding="utf-8")

    print(f"\nArtefak tersimpan di: {out_dir.resolve()}")
    print("  rupiah_final.keras      (output logits, untuk kalibrasi)")
    print("  rupiah_infer.keras      (output softmax, untuk ekspor)")
    print("  rupiah_mobilenetv2/     (SavedModel)")
    print("  test_logits.npz         (dipakai 03_calibrate_threshold.py)")
    print("\nLanjut:")
    print("  python scripts/03_calibrate_threshold.py --models models")
    print("  python scripts/02_export_tflite.py --model models/rupiah_infer.keras "
          "--data data/classification --output models/tflite")


if __name__ == "__main__":
    main()
