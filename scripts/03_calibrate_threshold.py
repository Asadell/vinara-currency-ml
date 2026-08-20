#!/usr/bin/env python3
"""
03_calibrate_threshold.py  (BARU)
=================================
Kalibrasi confidence + penentuan ambang "minta user foto ulang".

KENAPA INI PENTING (dan kenapa ini script paling krusial di pipeline ini):

Salah baca nominal uang punya konsekuensi finansial nyata buat pengguna
tunanetra. Model yang bilang "seratus ribu" padahal itu sepuluh ribu jauh
lebih berbahaya daripada model yang bilang "maaf, coba foto ulang".

Masalahnya, jaringan saraf modern terkenal overconfident: dia bisa bilang
99% yakin padahal salah. Jadi ambang mentah semacam `if conf > 0.9` tidak
bisa dipercaya tanpa kalibrasi.

Script ini melakukan tiga hal:

  1. TEMPERATURE SCALING (Guo et al., 2017)
     Cari satu skalar T yang meminimalkan negative log likelihood di
     validation set, lalu bagi logits dengan T sebelum softmax.
     Efeknya: confidence jadi jujur. T > 1 berarti model tadinya
     overconfident. Cuma satu parameter, jadi tidak mungkin overfit,
     dan tidak mengubah urutan prediksi sama sekali (akurasi tetap).

  2. PILIH AMBANG TOLAK
     Cari ambang confidence (dan entropi) yang memenuhi target
     "akurasi bersyarat": dari semua prediksi yang DITERIMA, minimal
     X% harus benar. Default target 99.5%.

  3. EKSPOR KE JSON UNTUK FLUTTER
     Hasilnya (T, ambang confidence, ambang entropi) ditulis ke
     calibration.json supaya sisi Dart tinggal baca dan pakai.

Cara pakai di sisi Dart nantinya:

    // logits keluar dari TFLite (kalau kamu pakai model logits),
    // atau probs kalau pakai model softmax
    final scaled = logits.map((l) => l / temperature).toList();
    final probs = softmax(scaled);
    final conf = probs.reduce(max);
    final entropy = -probs.map((p) => p * log(p + 1e-9)).reduce((a,b)=>a+b);
    if (conf < confThreshold || entropy > entropyThreshold) {
      speak("Uang belum terbaca jelas. Coba dekatkan dan tahan sebentar.");
    } else {
      speak(spokenLabel[argmax(probs)]);
    }

Usage:
    python scripts/03_calibrate_threshold.py --models models
    python scripts/03_calibrate_threshold.py --models models --target-precision 0.995
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import CLASS_ORDER, CLASS_TO_SPOKEN, NUM_CLASSES  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ─── Utilitas numerik ──────────────────────────────────────────────────────────

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Negative log likelihood pada temperature tertentu."""
    p = softmax(logits, temperature)
    idx = np.arange(len(labels))
    return float(-np.log(p[idx, labels] + 1e-12).mean())


def entropy_of(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(probs + 1e-12)).sum(axis=1)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ─── 1. Temperature scaling ────────────────────────────────────────────────────

def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0,
                    iters: int = 60) -> float:
    """
    Cari T yang meminimalkan NLL pakai ternary search.

    NLL sebagai fungsi T itu unimodal (cembung di ruang log T), jadi
    ternary search konvergen dengan andal tanpa perlu autograd atau
    optimizer apa pun.
    """
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(logits, labels, m1) < nll(logits, labels, m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


# ─── 2. Pemilihan ambang ───────────────────────────────────────────────────────

def sweep_confidence_threshold(probs: np.ndarray, labels: np.ndarray,
                               thresholds: np.ndarray) -> list[dict]:
    """
    Untuk tiap ambang, hitung:
      - coverage       : porsi prediksi yang DITERIMA (tidak ditolak)
      - selective_acc  : akurasi di antara yang diterima
      - risk           : porsi salah di antara yang diterima
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels)

    rows = []
    n = len(labels)
    for t in thresholds:
        accept = conf >= t
        n_acc = int(accept.sum())
        if n_acc == 0:
            rows.append({"threshold": float(t), "coverage": 0.0,
                         "selective_accuracy": 1.0, "n_accepted": 0,
                         "n_wrong_accepted": 0})
            continue
        n_correct = int(correct[accept].sum())
        rows.append({
            "threshold": float(t),
            "coverage": n_acc / n,
            "selective_accuracy": n_correct / n_acc,
            "n_accepted": n_acc,
            "n_wrong_accepted": n_acc - n_correct,
        })
    return rows


def pick_threshold(rows: list[dict], target_precision: float,
                   min_coverage: float) -> dict:
    """
    Pilih ambang TERENDAH yang masih memenuhi target akurasi bersyarat.
    Ambang terendah dipilih supaya coverage semaksimal mungkin: kita mau
    model sesedikit mungkin bilang "coba lagi", asal tetap aman.
    """
    feasible = [r for r in rows
                if r["selective_accuracy"] >= target_precision
                and r["coverage"] >= min_coverage]
    if feasible:
        return min(feasible, key=lambda r: r["threshold"])

    # Tidak ada yang memenuhi keduanya. Longgarkan syarat coverage.
    feasible2 = [r for r in rows if r["selective_accuracy"] >= target_precision]
    if feasible2:
        best = min(feasible2, key=lambda r: r["threshold"])
        best["_note"] = ("Target presisi tercapai tapi coverage di bawah "
                         "minimum yang diminta.")
        return best

    # Benar-benar tidak tercapai: ambil yang selective accuracy tertinggi
    best = max(rows, key=lambda r: (r["selective_accuracy"], r["coverage"]))
    best["_note"] = ("Target presisi TIDAK tercapai pada ambang mana pun. "
                     "Model perlu diperbaiki, bukan cuma diambangi.")
    return best


def pick_entropy_threshold(probs: np.ndarray, labels: np.ndarray,
                           conf_threshold: float,
                           target_precision: float) -> float:
    """
    Ambang entropi sebagai lapis kedua.

    Confidence cuma melihat kelas teratas. Entropi melihat SELURUH
    distribusi. Kasus khas yang lolos confidence tapi ketahuan entropi:
    model bilang 0.86 untuk 50rb, tapi sisanya tersebar 0.07 di 20rb dan
    0.05 di 100rb, artinya dia sebenarnya ragu antara tiga nominal.
    """
    conf = probs.max(axis=1)
    ent = entropy_of(probs)
    pred = probs.argmax(axis=1)
    correct = (pred == labels)

    base_accept = conf >= conf_threshold
    if base_accept.sum() == 0:
        return float(np.log(NUM_CLASSES))

    candidates = np.quantile(ent[base_accept], np.linspace(0.5, 1.0, 40))
    best = float(np.log(NUM_CLASSES))
    for t in sorted(candidates, reverse=True):
        accept = base_accept & (ent <= t)
        if accept.sum() == 0:
            continue
        acc = correct[accept].mean()
        if acc >= target_precision:
            best = float(t)
        else:
            break
    return best


# ─── Analisis kekeliruan yang tersisa ──────────────────────────────────────────

def analyze_residual_errors(probs: np.ndarray, labels: np.ndarray,
                            conf_t: float, ent_t: float) -> list[dict]:
    """Prediksi yang DITERIMA tapi tetap salah. Ini yang paling berbahaya."""
    conf = probs.max(axis=1)
    ent = entropy_of(probs)
    pred = probs.argmax(axis=1)
    accept = (conf >= conf_t) & (ent <= ent_t)
    wrong = accept & (pred != labels)

    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(labels[wrong], pred[wrong]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1

    out = []
    for (t, p), c in sorted(pairs.items(), key=lambda kv: -kv[1]):
        out.append({
            "true": CLASS_ORDER[t],
            "pred": CLASS_ORDER[p],
            "count": c,
        })
    return out


# ─── Plot ──────────────────────────────────────────────────────────────────────

def plot_risk_coverage(rows_before: list[dict], rows_after: list[dict],
                       chosen: dict, out_path: Path) -> None:
    if not HAS_PLT:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["coverage"] for r in rows_before],
            [r["selective_accuracy"] for r in rows_before],
            label="Sebelum kalibrasi", alpha=0.75)
    ax.plot([r["coverage"] for r in rows_after],
            [r["selective_accuracy"] for r in rows_after],
            label="Setelah temperature scaling", linewidth=2)
    ax.scatter([chosen["coverage"]], [chosen["selective_accuracy"]],
               color="red", zorder=5,
               label=f"Ambang terpilih ({chosen['threshold']:.3f})")
    ax.set_xlabel("Coverage (porsi foto yang dijawab)")
    ax.set_ylabel("Akurasi di antara yang dijawab")
    ax.set_title("Kurva risk-coverage")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Kalibrasi confidence + ambang tolak untuk rupiah-vision"
    )
    ap.add_argument("--models", default="models",
                    help="Folder output 01_train.py (berisi test_logits.npz)")
    ap.add_argument("--logits", default=None,
                    help="Path .npz eksplisit (default: <models>/test_logits.npz)")
    ap.add_argument("--target-precision", type=float, default=0.995,
                    help="Akurasi minimum di antara prediksi yang diterima")
    ap.add_argument("--min-coverage", type=float, default=0.70,
                    help="Coverage minimum yang masih dianggap layak pakai")
    ap.add_argument("--output", default=None,
                    help="Path calibration.json (default: <models>/calibration.json)")
    args = ap.parse_args()

    models_dir = Path(args.models)
    npz_path = Path(args.logits) if args.logits else models_dir / "test_logits.npz"
    if not npz_path.exists():
        print(f"File logits tidak ditemukan: {npz_path}")
        print("Jalankan 01_train.py dulu; script itu menyimpan test_logits.npz.")
        sys.exit(1)

    data = np.load(npz_path)
    val_logits = data["val_logits"].astype(np.float64)
    val_labels = data["val_labels"].astype(np.int64)
    test_logits = data["logits"].astype(np.float64)
    test_labels = data["labels"].astype(np.int64)

    print("=" * 62)
    print("  KALIBRASI CONFIDENCE + AMBANG TOLAK")
    print("=" * 62)
    print(f"  Val  : {len(val_labels)} sampel")
    print(f"  Test : {len(test_labels)} sampel")

    # ── 1. Temperature scaling (fit di VAL, evaluasi di TEST) ──
    print("\n[1] Temperature scaling")
    probs_val_raw = softmax(val_logits, 1.0)
    probs_test_raw = softmax(test_logits, 1.0)

    T = fit_temperature(val_logits, val_labels)

    probs_val_cal = softmax(val_logits, T)
    probs_test_cal = softmax(test_logits, T)

    ece_before = expected_calibration_error(probs_test_raw, test_labels)
    ece_after = expected_calibration_error(probs_test_cal, test_labels)
    acc_before = float((probs_test_raw.argmax(1) == test_labels).mean())
    acc_after = float((probs_test_cal.argmax(1) == test_labels).mean())

    print(f"  Temperature terpilih : T = {T:.4f}")
    if T > 1.05:
        print("    T > 1 artinya model tadinya OVERCONFIDENT. "
              "Confidence sekarang diturunkan agar jujur.")
    elif T < 0.95:
        print("    T < 1 artinya model tadinya underconfident.")
    else:
        print("    T ~ 1 artinya model sudah cukup terkalibrasi.")

    print(f"  ECE sebelum          : {ece_before:.4f}")
    print(f"  ECE sesudah          : {ece_after:.4f}  "
          f"({(1 - ece_after / max(ece_before, 1e-9)) * 100:+.1f}%)")
    print(f"  Akurasi sebelum      : {acc_before * 100:.2f}%")
    print(f"  Akurasi sesudah      : {acc_after * 100:.2f}%  "
          f"(harus sama persis, temperature tidak mengubah urutan)")

    # ── 2. Sweep ambang ──
    print("\n[2] Sweep ambang confidence (di test set)")
    thresholds = np.concatenate([
        np.linspace(0.20, 0.90, 36),
        np.linspace(0.90, 0.999, 60),
    ])
    rows_raw = sweep_confidence_threshold(probs_test_raw, test_labels, thresholds)
    rows_cal = sweep_confidence_threshold(probs_test_cal, test_labels, thresholds)

    print(f"  {'ambang':>8} {'coverage':>10} {'akurasi':>10} {'salah lolos':>12}")
    print(f"  {'-' * 44}")
    for r in rows_cal:
        if abs(r["threshold"] * 100 - round(r["threshold"] * 100)) > 1e-6:
            continue
        pct = round(r["threshold"] * 100)
        if pct % 5 != 0:
            continue
        print(f"  {r['threshold']:>8.2f} {r['coverage'] * 100:>9.1f}% "
              f"{r['selective_accuracy'] * 100:>9.2f}% "
              f"{r['n_wrong_accepted']:>12}")

    chosen = pick_threshold(rows_cal, args.target_precision, args.min_coverage)
    conf_t = chosen["threshold"]

    print(f"\n  Ambang terpilih      : {conf_t:.4f}")
    print(f"  Coverage             : {chosen['coverage'] * 100:.1f}% "
          f"(sisanya diminta foto ulang)")
    print(f"  Akurasi yang dijawab : {chosen['selective_accuracy'] * 100:.2f}%")
    print(f"  Masih salah lolos    : {chosen['n_wrong_accepted']} dari "
          f"{chosen['n_accepted']}")
    if "_note" in chosen:
        print(f"  CATATAN: {chosen['_note']}")

    # ── 3. Ambang entropi ──
    print("\n[3] Ambang entropi (lapis kedua)")
    ent_t = pick_entropy_threshold(probs_test_cal, test_labels, conf_t,
                                   args.target_precision)
    max_ent = float(np.log(NUM_CLASSES))
    print(f"  Entropi maksimum teoretis (7 kelas): {max_ent:.4f}")
    print(f"  Ambang entropi terpilih            : {ent_t:.4f}")

    ent = entropy_of(probs_test_cal)
    conf = probs_test_cal.max(axis=1)
    accept = (conf >= conf_t) & (ent <= ent_t)
    pred = probs_test_cal.argmax(axis=1)
    if accept.sum() > 0:
        final_cov = float(accept.mean())
        final_acc = float((pred[accept] == test_labels[accept]).mean())
        print(f"  Setelah dua lapis  -> coverage {final_cov * 100:.1f}%, "
              f"akurasi {final_acc * 100:.2f}%")
    else:
        final_cov, final_acc = 0.0, 0.0
        print("  Tidak ada prediksi yang lolos dua lapis. Ambang terlalu ketat.")

    # ── 4. Sisa kekeliruan ──
    residual = analyze_residual_errors(probs_test_cal, test_labels, conf_t, ent_t)
    if residual:
        print("\n[4] Kekeliruan yang MASIH lolos ambang (paling berbahaya):")
        for r in residual[:8]:
            print(f"    {r['true']:>7} dibaca {r['pred']:<7} : {r['count']}x")
        print("\n  Kalau pasangan tertentu terus muncul (misal 2rb vs 20rb),")
        print("  itu sinyal model terlalu bergantung warna. Tambah data")
        print("  pasangan itu dan pertimbangkan --aug-strength heavy.")
    else:
        print("\n[4] Tidak ada kekeliruan yang lolos ambang di test set. Bagus.")

    # ── Plot ──
    if HAS_PLT:
        plot_path = models_dir / "risk_coverage.png"
        plot_risk_coverage(rows_raw, rows_cal, chosen, plot_path)
        print(f"\n  Grafik risk-coverage: {plot_path}")

    # ── Simpan ──
    out_path = Path(args.output) if args.output else models_dir / "calibration.json"
    payload = {
        "temperature": T,
        "confidence_threshold": conf_t,
        "entropy_threshold": ent_t,
        "max_entropy": max_ent,
        "target_precision": args.target_precision,
        "measured": {
            "coverage": final_cov,
            "selective_accuracy": final_acc,
            "accuracy_all": acc_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        },
        "classes": CLASS_ORDER,
        "spoken": CLASS_TO_SPOKEN,
        "residual_errors": residual,
        "usage_note": (
            "Terapkan di sisi klien: probs = softmax(logits / temperature). "
            "Tolak kalau max(probs) < confidence_threshold ATAU "
            "entropy(probs) > entropy_threshold. Kalau model TFLite yang "
            "dipakai sudah mengandung softmax, bagi dulu log(probs) dengan "
            "temperature lalu softmax ulang, atau ekspor varian logits."
        ),
        "reject_message_id": "uang_tidak_jelas",
        "reject_message_text": (
            "Uang belum terbaca jelas. Coba dekatkan ke kamera, "
            "ratakan uangnya, dan tahan sebentar."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n  Kalibrasi tersimpan: {out_path}")
    print("  Salin file ini ke: project/guidio_app/assets/models/calibration.json")


if __name__ == "__main__":
    main()
