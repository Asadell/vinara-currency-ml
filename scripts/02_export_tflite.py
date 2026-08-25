#!/usr/bin/env python3
"""
02_export_tflite.py  (REVISI)
=============================
Ekspor model Keras -> TFLite (FP32 / FP16 / INT8-float-IO / INT8-full).

PERBAIKAN PENTING DARI VERSI LAMA:

  1. REPRESENTATIVE DATASET PAKAI PREPROCESSING YANG SAMA DENGAN TRAINING
     Versi lama memakai `tf.image.resize` biasa (squash ke persegi),
     sementara training barunya memakai letterbox. Kalau kalibrasi INT8
     melihat distribusi pixel yang berbeda dari yang dilihat model saat
     training, rentang quantization-nya meleset dan akurasi INT8 anjlok
     tanpa alasan jelas. Sekarang keduanya memakai fungsi yang sama
     dari src/data.py.

  2. SAMPEL KALIBRASI DIAMBIL SEIMBANG PER KELAS
     Versi lama mengambil sampel acak dari seluruh train set, jadi kelas
     mayoritas mendominasi kalibrasi. Sekarang tiap kelas menyumbang
     jumlah sampel yang sama.

  3. DUA VARIAN INT8
     - int8_floatio : bobot & aktivasi INT8, tapi input/output tetap
                      float32. Ini yang paling gampang dipakai di Flutter
                      karena kamu tinggal kirim float [-1,1] seperti biasa,
                      tanpa perlu urus zero_point/scale. REKOMENDASI DEPLOY.
     - int8_full    : input & output ikut INT8. Ukuran mirip, tapi wajib
                      quantize manual di sisi Dart. Berguna kalau kamu mau
                      pakai delegate NNAPI/EdgeTPU yang menuntut full-int.

  4. VALIDASI RENTANG INPUT
     Script memverifikasi bahwa model benar-benar mengharapkan input
     [-1, 1] dan mencetak parameter quantization (scale, zero_point)
     supaya bisa disalin ke kode Flutter tanpa tebak-tebakan.

  5. PERBANDINGAN AKURASI PER KELAS FP32 vs INT8
     Bukan cuma akurasi agregat. Kalau INT8 menjatuhkan satu kelas saja
     secara drastis (misal 2rb vs 20rb jadi ketuker), itu ketahuan di sini.

Usage:
    python scripts/02_export_tflite.py \
        --model models/rupiah_infer.keras \
        --data data/classification \
        --output models/tflite \
        --rep-samples 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
tf.keras.mixed_precision.set_global_policy("float32")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import CLASS_ORDER, NUM_CLASSES  # noqa: E402
from src.data import (  # noqa: E402
    decode_uint8,
    letterbox_and_normalize,
    scan_split,
    scan_split_per_class,
)


# ─── Representative dataset ────────────────────────────────────────────────────

def make_representative_dataset(data_dir: Path, img_size: int,
                                n_samples: int, seed: int = 7):
    """
    Generator kalibrasi INT8.

    Sampel diambil SEIMBANG per kelas dan diproses dengan letterbox +
    normalisasi yang identik dengan training.
    """
    rng = random.Random(seed)
    per_class = scan_split_per_class(data_dir, "train")

    n_per_class = max(1, n_samples // max(1, sum(1 for p in per_class if p)))
    selected: list[str] = []
    for paths in per_class:
        if not paths:
            continue
        pool = list(paths)
        rng.shuffle(pool)
        selected.extend(pool[:n_per_class])
    rng.shuffle(selected)
    selected = selected[:n_samples]

    print(f"   Representative dataset: {len(selected)} gambar "
          f"(~{n_per_class} per kelas)")

    def generator():
        for p in selected:
            img = decode_uint8(tf.constant(p))
            img = letterbox_and_normalize(img, img_size)
            yield [tf.expand_dims(img, 0)]

    return generator


# ─── Converter helper ──────────────────────────────────────────────────────────

def load_any_model(path: str) -> tf.keras.Model:
    """Muat .keras, SavedModel, atau folder export."""
    tf.keras.mixed_precision.set_global_policy("float32")
    p = Path(path)
    if p.suffix == ".keras" or p.suffix == ".h5":
        trained_model = tf.keras.models.load_model(str(p), compile=False)
        try:
            base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3), include_top=False, weights=None, alpha=1.0
            )
            inputs = tf.keras.Input(shape=(224, 224, 3), name="image", dtype="float32")
            x = base(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
            x = tf.keras.layers.Dropout(0.35, name="drop1")(x)
            x = tf.keras.layers.Dense(128, use_bias=False, name="fc1")(x)
            x = tf.keras.layers.BatchNormalization(name="fc1_bn")(x)
            x = tf.keras.layers.Activation("relu", name="fc1_relu")(x)
            x = tf.keras.layers.Dropout(0.35 * 0.6, name="drop2")(x)
            logits = tf.keras.layers.Dense(NUM_CLASSES, name="logits")(x)
            softmax_out = tf.keras.layers.Activation("softmax", name="probs", dtype="float32")(logits)
            float32_model = tf.keras.Model(inputs=inputs, outputs=softmax_out, name="rupiah_infer_float32")
            float32_model.set_weights(trained_model.get_weights())
            print("   ✅ Berhasil merekonstruksi model ke float32 murni untuk TFLite converter.")
            return float32_model
        except Exception as e:
            print(f"   (rekonstruksi float32 gagal: {e}; memakai model mentah)")
            return trained_model
    return tf.keras.layers.TFSMLayer(str(p), call_endpoint="serve")


def build_converter(model, tmp_dir: str):
    """Bikin converter dari model Keras."""
    tf.keras.mixed_precision.set_global_policy("float32")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        return tf.lite.TFLiteConverter.from_keras_model(model)
    except Exception as e:
        print(f"   (from_keras_model gagal: {e}; pakai SavedModel)")
        model.export(tmp_dir)
        return tf.lite.TFLiteConverter.from_saved_model(tmp_dir)


def write_model(tflite_bytes: bytes, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tflite_bytes)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"   Tersimpan: {path.name} ({size_mb:.2f} MB)")
    return size_mb


def export_fp32(model, path: Path) -> float:
    print("\n[1/4] Ekspor FP32")
    conv = build_converter(model, "/tmp/rv_sm_fp32")
    return write_model(conv.convert(), path)


def export_fp16(model, path: Path) -> float:
    print("\n[2/4] Ekspor FP16")
    conv = build_converter(model, "/tmp/rv_sm_fp16")
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return write_model(conv.convert(), path)


def export_int8_float_io(model, path: Path, rep_gen) -> float:
    print("\n[3/4] Ekspor INT8 (bobot+aktivasi INT8, I/O float32)")
    conv = build_converter(model, "/tmp/rv_sm_int8f")
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # I/O sengaja dibiarkan float32
    return write_model(conv.convert(), path)


def export_int8_full(model, path: Path, rep_gen) -> float:
    print("\n[4/4] Ekspor INT8 penuh (I/O ikut INT8)")
    conv = build_converter(model, "/tmp/rv_sm_int8u")
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return write_model(conv.convert(), path)


# ─── Inspeksi & verifikasi ─────────────────────────────────────────────────────

def inspect_tflite(path: Path) -> dict:
    """Cetak detail input/output termasuk parameter quantization."""
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    info = {
        "input": {
            "dtype": str(np.dtype(inp["dtype"])),
            "shape": [int(x) for x in inp["shape"]],
            "quantization": [float(inp["quantization"][0]),
                             int(inp["quantization"][1])],
        },
        "output": {
            "dtype": str(np.dtype(out["dtype"])),
            "shape": [int(x) for x in out["shape"]],
            "quantization": [float(out["quantization"][0]),
                             int(out["quantization"][1])],
        },
    }
    print(f"   input : {info['input']['dtype']} {info['input']['shape']} "
          f"quant(scale={info['input']['quantization'][0]:.6f}, "
          f"zero={info['input']['quantization'][1]})")
    print(f"   output: {info['output']['dtype']} {info['output']['shape']} "
          f"quant(scale={info['output']['quantization'][0]:.6f}, "
          f"zero={info['output']['quantization'][1]})")
    return info


def run_tflite_eval(path: Path, data_dir: Path, split: str,
                    img_size: int, max_images: int | None = None) -> dict:
    """
    Jalankan inferensi TFLite pada satu split, hitung akurasi total
    dan per kelas. Menangani input float32 maupun int8 otomatis.
    """
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_dtype = np.dtype(inp["dtype"])
    in_scale, in_zero = inp["quantization"]
    out_dtype = np.dtype(out["dtype"])
    out_scale, out_zero = out["quantization"]

    paths, labels = scan_split(data_dir, split)
    if max_images is not None and len(paths) > max_images:
        idx = np.random.RandomState(0).choice(len(paths), max_images,
                                              replace=False)
        paths = [paths[i] for i in idx]
        labels = [labels[i] for i in idx]

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for p, y in zip(paths, labels):
        img = decode_uint8(tf.constant(p))
        img = letterbox_and_normalize(img, img_size).numpy()
        x = np.expand_dims(img, 0)

        if in_dtype == np.int8:
            x = np.round(x / in_scale + in_zero).astype(np.int8)
        elif in_dtype == np.uint8:
            x = np.round(x / in_scale + in_zero).astype(np.uint8)
        else:
            x = x.astype(np.float32)

        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0]

        if out_dtype in (np.int8, np.uint8):
            probs = (raw.astype(np.float32) - out_zero) * out_scale
        else:
            probs = raw.astype(np.float32)

        cm[y, int(np.argmax(probs))] += 1

    total = cm.sum()
    correct = np.trace(cm)
    acc = float(correct / total) if total else 0.0

    per_class = {}
    for i, cls in enumerate(CLASS_ORDER):
        sup = int(cm[i].sum())
        rec = float(cm[i, i] / sup) if sup else 0.0
        per_class[cls] = {"support": sup, "recall": rec}

    return {"accuracy": acc, "per_class": per_class,
            "confusion_matrix": cm.tolist(), "n_images": int(total)}


def verify_input_contract(model, img_size: int) -> None:
    """
    Verifikasi kontrak input: model harus menerima [-1, 1] dan
    keluaran softmax harus menjumlah ke 1.
    """
    print("\nVerifikasi kontrak input/output model Keras:")
    probe_min = np.full((1, img_size, img_size, 3), -1.0, dtype=np.float32)
    probe_max = np.full((1, img_size, img_size, 3), 1.0, dtype=np.float32)

    y_min = np.asarray(model(probe_min, training=False))
    y_max = np.asarray(model(probe_max, training=False))

    print(f"   Output shape        : {y_min.shape}")
    print(f"   Jumlah prob (min in): {y_min.sum():.4f}")
    print(f"   Jumlah prob (max in): {y_max.sum():.4f}")

    if abs(y_min.sum() - 1.0) > 1e-2:
        print("   PERINGATAN: output tidak menjumlah 1. Pastikan kamu "
              "mengekspor models/rupiah_infer.keras (yang ada softmax-nya), "
              "bukan rupiah_final.keras (logits).")
    else:
        print("   OK: output berupa distribusi probabilitas.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ekspor Keras -> TFLite")
    ap.add_argument("--model", default="models/rupiah_infer.keras",
                    help="Model dengan output SOFTMAX (bukan logits)")
    ap.add_argument("--data", default="data/classification")
    ap.add_argument("--output", default="models/tflite")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--rep-samples", type=int, default=400)
    ap.add_argument("--eval-max", type=int, default=400,
                    help="Batas gambar untuk verifikasi akurasi tiap varian")
    ap.add_argument("--split", default="test",
                    help="Split yang dipakai verifikasi (test/val)")
    ap.add_argument("--skip-full-int8", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow {tf.__version__}")
    print(f"Memuat model: {args.model}")
    model = load_any_model(args.model)
    verify_input_contract(model, args.img_size)

    paths = {
        "fp32": out_dir / "rupiah_classifier_fp32.tflite",
        "fp16": out_dir / "rupiah_classifier_fp16.tflite",
        "int8_floatio": out_dir / "rupiah_classifier_int8.tflite",
        "int8_full": out_dir / "rupiah_classifier_int8_full.tflite",
    }

    sizes: dict[str, float] = {}
    sizes["fp32"] = export_fp32(model, paths["fp32"])
    sizes["fp16"] = export_fp16(model, paths["fp16"])

    rep_gen = make_representative_dataset(data_dir, args.img_size,
                                          args.rep_samples)
    sizes["int8_floatio"] = export_int8_float_io(model, paths["int8_floatio"],
                                                 rep_gen)
    if not args.skip_full_int8:
        rep_gen2 = make_representative_dataset(data_dir, args.img_size,
                                               args.rep_samples)
        sizes["int8_full"] = export_int8_full(model, paths["int8_full"],
                                              rep_gen2)

    # ── Inspeksi ──
    print("\n" + "=" * 62)
    print("  DETAIL TENSOR (salin ini ke kode Flutter)")
    print("=" * 62)
    infos = {}
    for name, p in paths.items():
        if not p.exists():
            continue
        print(f"\n {name}:")
        infos[name] = inspect_tflite(p)

    # ── Verifikasi akurasi ──
    evals: dict[str, dict] = {}
    if not args.no_verify:
        print("\n" + "=" * 62)
        print(f"  VERIFIKASI AKURASI (split: {args.split})")
        print("=" * 62)
        for name, p in paths.items():
            if not p.exists():
                continue
            print(f"\n {name} ...")
            ev = run_tflite_eval(p, data_dir, args.split, args.img_size,
                                 args.eval_max)
            evals[name] = ev
            print(f"   Akurasi: {ev['accuracy'] * 100:.2f}% "
                  f"({ev['n_images']} gambar)")

        if "fp32" in evals and "int8_floatio" in evals:
            print("\n  Recall per kelas, FP32 vs INT8:")
            print(f"  {'Kelas':>9} {'FP32':>8} {'INT8':>8} {'selisih':>9}")
            print(f"  {'-' * 38}")
            worst = 0.0
            worst_cls = None
            for cls in CLASS_ORDER:
                a = evals["fp32"]["per_class"][cls]["recall"]
                b = evals["int8_floatio"]["per_class"][cls]["recall"]
                d = b - a
                if d < worst:
                    worst, worst_cls = d, cls
                print(f"  {cls:>9} {a * 100:>7.1f}% {b * 100:>7.1f}% "
                      f"{d * 100:>8.1f}")
            drop = evals["fp32"]["accuracy"] - evals["int8_floatio"]["accuracy"]
            print(f"\n  Penurunan akurasi total akibat INT8: {drop * 100:.2f} poin")
            if drop > 0.02:
                print("  Penurunan lebih dari 2 poin. Coba naikkan "
                      "--rep-samples, atau pakai FP16 kalau ukuran masih muat.")
            if worst_cls and worst < -0.05:
                print(f"  Perhatian: kelas {worst_cls} turun "
                      f"{abs(worst) * 100:.1f} poin. Periksa apakah kelas ini "
                      f"kurang terwakili di representative dataset.")

    # ── Salin label ──
    for src_name in ("labels.txt",):
        for src in (data_dir / src_name, Path("models") / src_name):
            if src.exists():
                shutil.copy(src, out_dir / src_name)
                break

    # ── Ringkasan ──
    print("\n" + "=" * 62)
    print("  RINGKASAN EKSPOR")
    print("=" * 62)
    print(f"  {'Varian':<16} {'Ukuran':>9} {'Akurasi':>9}")
    print(f"  {'-' * 38}")
    for name in ("fp32", "fp16", "int8_floatio", "int8_full"):
        if name not in sizes:
            continue
        acc = evals.get(name, {}).get("accuracy")
        acc_s = f"{acc * 100:.2f}%" if acc is not None else "N/A"
        print(f"  {name:<16} {sizes[name]:>7.2f}MB {acc_s:>9}")

    summary = {
        "sizes_mb": sizes,
        "tensor_info": infos,
        "evaluation": evals,
        "img_size": args.img_size,
        "preprocessing_contract": {
            "resize": "letterbox: resize jaga aspect ratio lalu pad 0 di tengah",
            "normalization": "x / 127.5 - 1.0",
            "channel_order": "RGB",
            "recommended_for_flutter": "rupiah_classifier_int8.tflite (I/O float32)",
        },
        "classes": CLASS_ORDER,
    }
    with open(out_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Rekomendasi deploy Flutter: {paths['int8_floatio'].name}")
    print("  Alasan: bobot INT8 (kecil & cepat) tapi I/O tetap float32, "
          "jadi kode Dart tidak perlu urus scale/zero_point.")
    print(f"\n  Salin ke: project/guidio_app/assets/models/uang_rupiah.tflite")
    print(f"  Semua file: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
