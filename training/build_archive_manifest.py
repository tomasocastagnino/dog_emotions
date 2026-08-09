"""
build_archive_manifest.py -- Arma un manifest CSV desde archive/Final dog dataset/
(otro dataset de Kaggle que el usuario bajo aparte) para poder sumarlo al
entrenamiento con --extra-manifest.

Excluye cualquier imagen que ya este en el test set original (mismo nombre de
archivo) -- sin este chequeo se corre el riesgo de "entrenar con el examen":
~468 de las 3876 imagenes de este dataset resultaron ser las mismas que ya
estan en el 20% de test que separamos desde el principio.

Uso:
    python training/build_archive_manifest.py
"""

import csv
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lightweight import CLASSES, get_dataset_dir, split_dataset  # noqa: E402

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "archive", "Final dog dataset", "Final dog dataset")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "data", "archive_manifest.csv")


def main():
    data_dir = get_dataset_dir()
    (_, _), (_, _), (p_test, y_test) = split_dataset(data_dir)
    test_basenames = set(os.path.basename(p) for p in p_test)

    rows = []
    skipped_test = 0
    per_class = {c: 0 for c in CLASSES}
    for cls in CLASSES:
        for f in sorted(pathlib.Path(ARCHIVE_DIR, cls).glob("*")):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if f.name in test_basenames:
                skipped_test += 1
                continue
            rows.append((str(f), cls))
            per_class[cls] += 1

    print(f"Excluidas por estar en el test set original: {skipped_test}")
    print(f"Imagenes utilizables: {len(rows)}")
    for c in CLASSES:
        print(f"  {c}: {per_class[c]}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    print(f"\nManifest guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
