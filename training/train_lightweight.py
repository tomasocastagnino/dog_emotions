"""
train_lightweight.py -- Entrena un modelo liviano (MobileNetV3Large o EfficientNetV2B0)
para clasificar emociones caninas (angry / happy / relaxed / sad), pensado para
terminar corriendo en el navegador via TensorFlow.js.

Es la continuacion de un trabajo practico universitario que entrenaba InceptionV3
para el mismo problema (76% accuracy, 124 MB -- repo original:
https://github.com/gaspigz/TP-FINAL-IIA). Este script es independiente de ese
notebook: descarga el dataset, arma los splits y entrena solo, sin depender de
ninguna celda previa.

Uso (correr siempre desde la raiz del repo):
    python training/train_lightweight.py --backbone mobilenetv3
    python training/train_lightweight.py --backbone efficientnetv2b0 --n-capas 30 --dropout 0.3

Funciona igual en Mac, Linux, Windows y Google Colab (si estas en Colab, subi este
archivo o clona el repo y corre `!python training/train_lightweight.py ...`).

Requiere las credenciales de Kaggle como en el proyecto original: `kaggle.json` en la
raiz del repo (local) o los Secrets KAGGLE_USERNAME/KAGGLE_KEY (Colab).
"""

import argparse
import csv
import json
import os
import pathlib
import pickle
import platform
import shutil
import tempfile
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
CLASSES = ["angry", "happy", "relaxed", "sad"]
AUTOTUNE = tf.data.AUTOTUNE

# MobileNetV3 y EfficientNetV2 traen el preprocesado (Rescaling) INCLUIDO adentro
# del modelo (include_preprocessing=True). A diferencia de InceptionV3, el dataset
# le tiene que entregar pixeles crudos en [0, 255], no en [-1, 1]. Por eso mas abajo
# NO se llama a ningun preprocess_input externo.
BACKBONES = {
    "mobilenetv3": lambda shape: keras.applications.MobileNetV3Large(
        include_top=False, weights="imagenet", input_shape=shape, include_preprocessing=True),
    "efficientnetv2b0": lambda shape: keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=shape, include_preprocessing=True),
}


# --- Dataset ------------------------------------------------------------------------
def get_dataset_dir() -> str:
    """Descarga el dataset de Kaggle (o reusa el cache local) y devuelve la carpeta
    con las 4 subcarpetas de clases. Detecta Colab vs local automaticamente, igual
    que el notebook original."""
    try:
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        print("Entorno detectado: Google Colab (credenciales desde Secrets)")
    except ImportError:
        cred_path = os.path.join(os.getcwd(), "kaggle.json")
        with open(cred_path) as f:
            creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
        print(f"Entorno detectado: local (credenciales desde {cred_path})")

    import kagglehub
    path = kagglehub.dataset_download("danielshanbalico/dog-emotion")
    return os.path.join(path, "Dog Emotion")


def list_files(data_dir: str):
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for f in sorted(pathlib.Path(data_dir, cls).glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(f))
                labels.append(idx)
    return np.array(paths), np.array(labels)


def split_dataset(data_dir: str):
    """Mismo split 60/20/20 estratificado y con la misma seed que el proyecto
    original -> el test_acc que da este script es comparable 1 a 1 contra el 0.760
    de InceptionV3."""
    paths_all, labels_all = list_files(data_dir)
    p_tmp, p_test, y_tmp, y_test = train_test_split(
        paths_all, labels_all, test_size=0.20, stratify=labels_all, random_state=SEED)
    p_train, p_val, y_train, y_val = train_test_split(
        p_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


def _load(p, y):
    img = tf.io.read_file(p)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return img, y


def make_dataset(paths, labels, cache_file, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=False)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(cache_file)
    if shuffle:
        ds = ds.shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


def load_extra_manifest(path, classes_filter):
    """Lee el manifest de training/filter_extra_images.py (path,label,confidence)
    y devuelve (paths, labels) limitado a las clases pedidas."""
    paths, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["label"] in classes_filter:
                paths.append(row["path"])
                labels.append(CLASSES.index(row["label"]))
    return np.array(paths), np.array(labels)


def _pil_load_resize(path_tensor):
    """Carga con PIL en vez de tf.io.decode_image. Mezclando Kaggle + Flickr,
    tf.io.decode_image tira "Unknown image file format" a mitad de
    entrenamiento de forma reproducible -- pasa igual con y sin cache, con y
    sin paralelismo, aunque cada archivo decodifica bien probado uno por uno
    con tf.io.decode_image fuera del pipeline. No se pudo aislar la causa exacta
    (huele a bug del decoder de TF con esta mezcla puntual de formatos/tamaños),
    asi que se lo evita del todo usando el loader de PIL que ya usa
    filter_extra_images.py para validar estas mismas imagenes."""
    path = path_tensor.numpy().decode("utf-8")
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    return tf.keras.utils.img_to_array(img).astype("float32")


def _load_weighted(p, y, w):
    img = tf.py_function(func=_pil_load_resize, inp=[p], Tout=tf.float32)
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, y, w


def make_weighted_dataset(paths, labels, weights, cache_file, shuffle=False):
    """Igual que make_dataset, pero con un peso por muestra (sample_weight) --
    para que los datos extra pesen menos en la loss que los del dataset curado."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_weighted, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


# --- Modelo ---------------------------------------------------------------------------
def unfreeze_last_n(base_model, n):
    """Descongela las ultimas n capas del backbone (sin tocar BatchNorm, para no
    perder las estadisticas de ImageNet con batches chicos de fine-tuning)."""
    if n <= 0:
        base_model.trainable = False
        return
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - n)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True


def build_model(backbone: str, augment: bool, dropout: float, n_unfrozen: int):
    base_model = BACKBONES[backbone]((*IMG_SIZE, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = inputs
    if augment:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="output")(x)
    model = keras.Model(inputs, outputs, name=f"{backbone}_dog_emotion")

    unfreeze_last_n(base_model, n_unfrozen)
    return model


# --- Memoria ---------------------------------------------------------------------------
def peak_memory_mb():
    """RAM pico (RSS) del proceso hasta este punto, en MB. None en Windows (el
    modulo `resource` no existe ahi -- no vale la pena instalar algo extra solo
    para esto, se puede mirar por Administrador de tareas)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss viene en bytes en macOS y en KB en Linux -- gotcha clasico del modulo.
        return round(peak / (1024 ** 2), 1) if platform.system() == "Darwin" else round(peak / 1024, 1)
    except ImportError:
        return None


# --- Entrenamiento ---------------------------------------------------------------------
def train(args):
    print(f"Backbone={args.backbone}  n_capas={args.n_capas}  dropout={args.dropout}  "
          f"augment={args.augment}  epochs<= {args.epochs}  patience={args.patience}")

    data_dir = get_dataset_dir()
    (p_train, y_train), (p_val, y_val), (p_test, y_test) = split_dataset(data_dir)
    print(f"train={len(p_train)}  val={len(p_val)}  test={len(p_test)} imagenes")

    # Datos extra (opcional): salen de training/filter_extra_images.py, que ya los
    # filtro contra el modelo actual -- aca solo se suman a TRAIN, nunca a val/test,
    # para no perder un punto de comparacion limpio contra las corridas anteriores.
    extra_info = None
    weights_train = np.ones(len(p_train), dtype="float32")
    if args.extra_manifest:
        extra_classes = [c.strip() for c in args.extra_classes.split(",")]
        p_extra, y_extra = load_extra_manifest(args.extra_manifest, extra_classes)
        print(f"Datos extra ({args.extra_manifest}): {len(p_extra)} imagenes de "
              f"{extra_classes}, peso={args.extra_weight}")
        p_train = np.concatenate([p_train, p_extra])
        y_train = np.concatenate([y_train, y_extra])
        weights_train = np.concatenate(
            [weights_train, np.full(len(p_extra), args.extra_weight, dtype="float32")])
        extra_info = {"manifest": args.extra_manifest, "classes": extra_classes,
                     "weight": args.extra_weight, "n_extra": int(len(p_extra))}
        print(f"train total con extra: {len(p_train)} imagenes")

    # Cache nuevo en cada corrida: si una corrida anterior lo dejo a medio escribir
    # (p. ej. por el chequeo interno que hace Keras al arrancar .fit()), la proxima
    # lo encuentra corrupto y tira el warning "did not fully read the dataset being
    # cached" -- entrena igual, pero recae a decodificar todo de nuevo sin avisar.
    # Empezar de cero por corrida evita ese estado compartido entre backbones.
    cache_dir = os.path.join(tempfile.gettempdir(), "dog_emotion_cache", f"{args.backbone}_{os.getpid()}")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    if extra_info:
        train_ds = make_weighted_dataset(p_train, y_train, weights_train,
                                         os.path.join(cache_dir, "train"), shuffle=True)
    else:
        train_ds = make_dataset(p_train, y_train, os.path.join(cache_dir, "train"), shuffle=True)
    val_ds = make_dataset(p_val, y_val, os.path.join(cache_dir, "val"))
    test_ds = make_dataset(p_test, y_test, os.path.join(cache_dir, "test"))

    base_layers = len(BACKBONES[args.backbone]((*IMG_SIZE, 3)).layers)
    print(f"{args.backbone} tiene {base_layers} capas en el backbone "
          f"(referencia para elegir --n-capas)")

    model = build_model(args.backbone, args.augment, args.dropout, args.n_capas)
    lr = 1e-3 if args.n_capas == 0 else 1e-4
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=args.patience, restore_best_weights=True)]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks)
    elapsed = time.time() - t0

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))

    extra_suffix = ""
    if extra_info:
        # el peso va en el nombre para no pisar corridas con el mismo backbone/clases
        # pero distinto --extra-weight (paso en el que ya nos pisamos una vez).
        extra_suffix = ("_extra" + "".join(c[:3] for c in extra_info["classes"]) +
                        f"_w{int(extra_info['weight'] * 100)}")
    run_name = (f"{args.backbone}_n{args.n_capas}_do{int(args.dropout * 100)}_"
               f"{'aug' if args.augment else 'noaug'}{extra_suffix}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("histories", exist_ok=True)
    model_path = f"models/{run_name}.keras"
    model.save(model_path)

    metadata = {
        "run_name": run_name,
        "backbone": args.backbone,
        "img_size": list(IMG_SIZE),
        "n_capas_descongeladas": args.n_capas,
        "dropout": args.dropout,
        "augment": args.augment,
        "epochs_corridas": len(history.history["loss"]),
        "segundos": round(elapsed, 1),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "trainable_params": trainable_params,
        "tamaño_mb": round(os.path.getsize(model_path) / 1024 / 1024, 2),
        "dataset": "danielshanbalico/dog-emotion (Kaggle)",
        "split_seed": SEED,
        "ram_pico_mb": peak_memory_mb(),
        "extra_data": extra_info,
    }
    with open(f"histories/{run_name}_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    with open(f"histories/{run_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n--- Resultado ---")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nModelo guardado en {model_path}")
    print(f"Referencia InceptionV3 (informe original): test_acc=0.760, tamaño=124.0 MB")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=list(BACKBONES), required=True,
                   help="Arquitectura liviana a entrenar")
    p.add_argument("--n-capas", type=int, default=20, dest="n_capas",
                   help="Cuantas capas del final del backbone descongelar (default: 20)")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Dropout antes de la capa densa final (default: 0.2)")
    p.add_argument("--augment", action="store_true", default=True,
                   help="Data augmentation: flip/rotacion/zoom (default: activado)")
    p.add_argument("--no-augment", action="store_false", dest="augment",
                   help="Desactiva la data augmentation")
    p.add_argument("--epochs", type=int, default=40, help="Tope de epocas (default: 40)")
    p.add_argument("--patience", type=int, default=6,
                   help="Paciencia de EarlyStopping sobre val_accuracy (default: 6)")
    p.add_argument("--extra-manifest", default=None,
                   help="CSV de training/filter_extra_images.py con datos extra ya filtrados")
    p.add_argument("--extra-classes", default="relaxed",
                   help="Clases (separadas por coma) para las que sumar datos extra (default: relaxed)")
    p.add_argument("--extra-weight", type=float, default=0.5,
                   help="Peso relativo de las imagenes extra en la loss (default: 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
