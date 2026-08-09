"""
convert_to_tfjs.py -- Convierte un modelo .keras (de train_lightweight.py) al
formato TensorFlow.js que usa docs/app.js.

Requiere el extra de requirements.txt: `pip install tensorflowjs "setuptools<81"`.

Por que no es un simple `tensorflowjs_converter model.keras docs/model`:
  1. Las capas de data augmentation (RandomFlip/RandomRotation/RandomZoom) tienen
     generadores de numeros aleatorios con estado que no se pueden "congelar" en
     un grafo estatico de inferencia -- y de todas formas no deben correr fuera
     de entrenamiento. Hay que reconstruir un modelo de inferencia sin ellas
     (mismos pesos, mismas capas de fondo, solo se saltea la augmentation).
  2. El conversor de tensorflowjs, pasando por el JSON de topologia de Keras 3
     directamente (`save_keras_model`), tira "InputLayer should be passed either
     a batchInputShape or an inputShape". La ruta que sí funciona es exportar
     primero como SavedModel (`model.export(...)`) y convertir ESO
     (`convert_tf_saved_model`), que da un "tfjs_graph_model" -- por eso
     docs/app.js usa `tf.loadGraphModel`, no `tf.loadLayersModel`.

Uso:
    python training/convert_to_tfjs.py --model models/mobilenetv3_n30_do20_aug.keras
"""

import argparse
import os
import shutil
import tempfile


def strip_augmentation(model):
    """Reconstruye un modelo de inferencia reusando las mismas capas entrenadas
    (mismos pesos), pero sin las capas de data augmentation del principio."""
    from tensorflow import keras

    base_model_layer = next(l for l in model.layers if isinstance(l, keras.Model))
    pool_layer = next(l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D))
    dense_layer = next(l for l in model.layers if isinstance(l, keras.layers.Dense))
    dropout_layer = next((l for l in model.layers if isinstance(l, keras.layers.Dropout)), None)

    input_shape = model.inputs[0].shape[1:]
    inputs = keras.Input(shape=input_shape)
    x = base_model_layer(inputs, training=False)
    x = pool_layer(x)
    if dropout_layer is not None:
        x = dropout_layer(x, training=False)
    outputs = dense_layer(x)
    return keras.Model(inputs, outputs, name=f"{model.name}_inference")


def convert(model_path: str, out_dir: str):
    import numpy as np
    import tensorflow as tf
    import tensorflowjs as tfjs

    print(f"Cargando {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    inference_model = strip_augmentation(model)

    # Chequeo: la version sin augmentation tiene que predecir exactamente igual
    # que la original (la augmentation nunca corre en inferencia de todas formas).
    z = np.random.RandomState(0).uniform(0, 255, (2, *model.inputs[0].shape[1:])).astype("float32")
    diff = float(np.abs(model.predict(z, verbose=0) - inference_model.predict(z, verbose=0)).max())
    print(f"Diferencia max vs modelo original (debe ser 0.0): {diff}")
    if diff > 1e-6:
        raise RuntimeError("La version sin augmentation no predice igual que la original -- revisar strip_augmentation().")

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = os.path.join(tmp, "saved_model")
        inference_model.export(saved_model_dir)

        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        tfjs.converters.convert_tf_saved_model(
            saved_model_dir, out_dir,
            quantization_dtype_map={"uint8": "*"},
        )

    total_mb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1024 / 1024
    print(f"\nListo -> {out_dir}/  ({total_mb:.2f} MB)")
    print("docs/app.js espera tf.loadGraphModel('model/model.json') -- ya esta asi por default.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path al .keras a convertir")
    p.add_argument("--out", default="docs/model", help="Carpeta de salida (default: docs/model)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.model, args.out)
