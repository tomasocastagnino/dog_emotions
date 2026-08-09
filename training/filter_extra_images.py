"""
filter_extra_images.py -- Usa el modelo ya entrenado para "curar" data/images/
(dataset extra sin revisar, recolectado de Flickr) antes de sumarlo al
entrenamiento.

Para cada imagen en data/images/<clase>/, corre el modelo elegido y se queda
solo con las que:
  1. El modelo predice la MISMA clase que indica la carpeta de origen (señal
     cruzada: la etiqueta de Flickr y el modelo coinciden), y
  2. Con confianza >= --min-conf en esa predicción.

Esto descarta de un saque tanto las mal etiquetadas como las visualmente
ambiguas, sin revisar las ~16.000 imágenes a mano. El resultado queda en un
manifest CSV para reusar sin volver a correr el modelo sobre todo el dataset
cada vez.

Uso:
    python training/filter_extra_images.py --model models/mobilenetv3_n30_do20_aug.keras
"""

import argparse
import csv
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lightweight import CLASSES, IMG_SIZE  # noqa: E402

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "images")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "images_filtered_manifest.csv")


def list_extra_files():
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for f in sorted(pathlib.Path(IMAGES_DIR, cls).glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(f))
                labels.append(idx)
    return paths, labels


def iter_batches(paths, labels, img_size, batch_size):
    """Carga y arma batches a mano (no con tf.data) para poder saltear archivos
    corruptos de a uno -- un dataset scrapeado de Flickr casi seguro tiene
    algunos.

    Valida con AMBOS decodificadores: PIL (via load_img, lo que se usa aca abajo
    para predecir) y tf.io.decode_image (lo que usa despues train_lightweight.py
    dentro de un tf.data pipeline). Hay archivos que PIL abre sin quejarse pero
    que tf.io.decode_image rechaza -- sin este chequeo, esos terminan en el
    manifest y revientan el entrenamiento mas adelante con un error de
    "Unknown image file format" que tf.data no puede saltear a mitad de un
    pipeline ya armado."""
    import tensorflow as tf

    batch_imgs, batch_paths, batch_labels = [], [], []
    for p, y in zip(paths, labels):
        try:
            raw = tf.io.read_file(p)
            tf.io.decode_image(raw, channels=3, expand_animations=False)
            img = tf.keras.utils.load_img(p, target_size=img_size)
            arr = tf.keras.utils.img_to_array(img)
        except Exception as e:
            print(f"[SKIP] {p}: {e}")
            continue
        batch_imgs.append(arr)
        batch_paths.append(p)
        batch_labels.append(y)
        if len(batch_imgs) == batch_size:
            yield np.stack(batch_imgs), batch_paths, batch_labels
            batch_imgs, batch_paths, batch_labels = [], [], []
    if batch_imgs:
        yield np.stack(batch_imgs), batch_paths, batch_labels


def filter_images(model_path, min_conf, batch_size=64):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    paths, labels = list_extra_files()
    print(f"Total imágenes en {IMAGES_DIR}: {len(paths)}")

    kept = []
    n_processed = 0
    per_class_kept = {c: 0 for c in CLASSES}
    per_class_total = {c: 0 for c in CLASSES}

    for imgs, b_paths, b_labels in iter_batches(paths, labels, IMG_SIZE, batch_size):
        probs = model.predict(imgs, verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        pred_conf = probs[np.arange(len(probs)), pred_idx]
        for p, y, pi, pc in zip(b_paths, b_labels, pred_idx, pred_conf):
            per_class_total[CLASSES[y]] += 1
            n_processed += 1
            if pi == y and pc >= min_conf:
                kept.append((p, y, float(pc)))
                per_class_kept[CLASSES[y]] += 1
        if n_processed % 1600 == 0:
            print(f"  ... {n_processed}/{len(paths)} procesadas")

    print(f"\nProcesadas: {n_processed}")
    print("Por clase (guardadas / total -- % que sobrevive el filtro):")
    for c in CLASSES:
        tot = per_class_total[c]
        pct = 100 * per_class_kept[c] / tot if tot else 0
        print(f"  {c:8s}: {per_class_kept[c]:5d} / {tot:5d}  ({pct:.1f}%)")
    print(f"  {'TOTAL':8s}: {len(kept):5d} / {n_processed:5d}  ({100*len(kept)/n_processed:.1f}%)")

    return kept


def save_manifest(kept, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "confidence"])
        for p, y, conf in kept:
            writer.writerow([p, CLASSES[y], f"{conf:.4f}"])
    print(f"\nManifest guardado en {path} ({len(kept)} filas)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path al .keras a usar como filtro")
    p.add_argument("--min-conf", type=float, default=0.8,
                   help="Confianza minima para aceptar una imagen (default: 0.8)")
    p.add_argument("--out", default=MANIFEST_PATH, help="Path del manifest CSV de salida")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kept = filter_images(args.model, args.min_conf)
    save_manifest(kept, args.out)
