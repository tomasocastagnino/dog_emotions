"""
convert_dog_gate.py -- Genera docs/model_gate/: un MobileNetV3Small pre-entrenado
en ImageNet, SIN fine-tuning, usado como filtro de "¿hay un perro en el frame?"
antes de correr el clasificador de emociones.

Como funciona el filtro: en el set de 1000 clases estandar de ImageNet, los
indices 151 a 268 son exactamente las 118 razas de perro (de "Chihuahua" a
"Mexican_hairless" -- verificado contra imagenet_class_index.json). Sumando la
probabilidad de esas 118 clases se obtiene una señal de "es un perro" mucho mas
robusta que mirar solo la clase top-1 (que puede repartirse entre varias razas
parecidas sin que ninguna domine). No hace falta entrenar nada: el modelo ya
sabe reconocer razas de perro de fabrica.

Uso:
    python training/convert_dog_gate.py
"""

import os
import shutil
import tempfile

DOG_INDEX_START = 151
DOG_INDEX_END = 268  # inclusive
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "model_gate")


def convert():
    import tensorflow as tf
    import tensorflowjs as tfjs
    from tensorflow import keras

    print("Cargando MobileNetV3Small (ImageNet, sin fine-tuning) ...")
    model = keras.applications.MobileNetV3Small(weights="imagenet", include_top=True)

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = os.path.join(tmp, "saved_model")
        model.export(saved_model_dir)

        if os.path.isdir(OUT_DIR):
            shutil.rmtree(OUT_DIR)
        os.makedirs(OUT_DIR, exist_ok=True)

        tfjs.converters.convert_tf_saved_model(
            saved_model_dir, OUT_DIR,
            quantization_dtype_map={"uint8": "*"},
        )

    total_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)) / 1024 / 1024
    print(f"\nListo -> {OUT_DIR}/  ({total_mb:.2f} MB)")
    print(f"Indices de razas de perro en el softmax de 1000 clases: {DOG_INDEX_START}-{DOG_INDEX_END}")


if __name__ == "__main__":
    convert()
