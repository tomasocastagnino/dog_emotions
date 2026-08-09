"""
generate_report_assets.py -- Genera los graficos de REPORT.md: comparacion de
corridas, curvas de entrenamiento y matriz de confusion del modelo elegido.

Uso (desde la raiz del repo, con el venv activo):
    python training/generate_report_assets.py --model models/mobilenetv3_n30_do20_aug.keras
"""

import argparse
import glob
import json
import os
import pickle
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lightweight import CLASSES, get_dataset_dir, make_dataset, split_dataset  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report_assets")
BASELINE = {"run_name": "InceptionV3 (original)", "test_acc": 0.760, "tamaño_mb": 124.0}


def plot_comparison():
    rows = [json.load(open(f)) for f in sorted(glob.glob("histories/*_metadata.json"))]
    rows.sort(key=lambda d: -d["test_acc"])
    rows.append(BASELINE)

    names = [r["run_name"] for r in rows]
    accs = [r["test_acc"] * 100 for r in rows]
    sizes = [r["tamaño_mb"] for r in rows]
    colors = ["#999999" if r is BASELINE else "#2f6fed" for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.barh(names, accs, color=colors)
    ax1.set_xlabel("Test accuracy (%)")
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    for i, v in enumerate(accs):
        ax1.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

    ax2.barh(names, sizes, color=colors)
    ax2.set_xlabel("Tamaño del modelo (MB)")
    ax2.invert_yaxis()
    ax2.set_yticklabels([])
    for i, v in enumerate(sizes):
        ax2.text(v + 2, i, f"{v:.1f} MB", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "comparacion_modelos.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Guardado", path)


def plot_training_curves(history_path, run_name):
    with open(history_path, "rb") as f:
        h = pickle.load(f)

    best_ep = int(np.argmax(h["val_accuracy"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].axvline(best_ep, ls="--", c="gray", alpha=0.6)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("época")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(h["accuracy"], label="train")
    axes[1].plot(h["val_accuracy"], label="val")
    axes[1].axvline(best_ep, ls="--", c="gray", alpha=0.6)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("época")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"{run_name} — mejor época (val_accuracy) marcada en línea punteada: {best_ep}")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "curvas_entrenamiento.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Guardado", path)


def plot_confusion_matrix(model_path, run_name):
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    data_dir = get_dataset_dir()
    (_, _), (_, _), (p_test, y_test) = split_dataset(data_dir)

    cache_dir = os.path.join(tempfile.gettempdir(), "dog_emotion_report_cache")
    os.makedirs(cache_dir, exist_ok=True)
    test_ds = make_dataset(p_test, y_test, os.path.join(cache_dir, "test"))

    model = tf.keras.models.load_model(model_path)
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=3)
    print(report)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"{run_name}\n\n{report}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"{run_name}\ntest_acc={(y_pred == y_test).mean():.4f}")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "matriz_confusion.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Guardado", path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path al .keras del modelo elegido")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(args.model))[0]
    history_path = f"histories/{run_name}_history.pkl"

    plot_comparison()
    plot_training_curves(history_path, run_name)
    plot_confusion_matrix(args.model, run_name)
