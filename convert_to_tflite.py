"""
convert_to_tflite.py — Convierte modelos .keras a .tflite.

El formato TFLite es mucho más liviano y rápido para inferencia en CPU,
sin necesidad de cargar TensorFlow completo en runtime.

Uso:
    python convert_to_tflite.py

Lista los .keras en models/, convertís el que elijas y guarda el .tflite
en el mismo directorio con el mismo nombre base.
"""

import os
import sys
import glob

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def list_keras_models() -> list[str]:
    return sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))


def choose(models: list[str]) -> str:
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Conversor .keras → .tflite                ║")
    print("╚══════════════════════════════════════════════╝\n")

    if not models:
        print(f"[ERROR] No hay modelos .keras en: {MODELS_DIR}")
        sys.exit(1)

    print("Modelos .keras disponibles:")
    for i, p in enumerate(models, 1):
        name = os.path.basename(p)
        size_mb = os.path.getsize(p) / 1024 / 1024
        tflite_path = p.replace(".keras", ".tflite")
        status = "ya convertido" if os.path.exists(tflite_path) else ""
        print(f"  [{i}] {name}  ({size_mb:.1f} MB)  {status}")

    print()
    while True:
        raw = input(f"Elegi un modelo [1-{len(models)}] (o 'todos' para convertir todos): ").strip()
        if raw.lower() == "todos":
            return "todos"
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        print(f"    Ingresa un numero entre 1 y {len(models)} o 'todos'.")


def convert(keras_path: str) -> str:
    """Convierte un .keras a .tflite con cuantizacion dinamica."""
    name = os.path.basename(keras_path)
    tflite_path = keras_path.replace(".keras", ".tflite")

    print(f"\n  Cargando {name} ...")
    model = tf.keras.models.load_model(keras_path)

    print(f"  Convirtiendo a TFLite ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Cuantizacion dinamica de pesos: reduce el modelo ~4x y acelera en CPU
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    original_mb = os.path.getsize(keras_path) / 1024 / 1024
    tflite_mb   = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"  Guardado: {os.path.basename(tflite_path)}")
    print(f"  Original: {original_mb:.1f} MB  ->  TFLite: {tflite_mb:.1f} MB  "
          f"(reduccion {(1 - tflite_mb/original_mb)*100:.0f}%)")
    return tflite_path


if __name__ == "__main__":
    models = list_keras_models()
    choice = choose(models)

    to_convert = models if choice == "todos" else [choice]

    print()
    for path in to_convert:
        convert(path)

    print("\nListo. Ahora podes usar app.py -- va a preferir los .tflite automaticamente.\n")
