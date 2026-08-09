"""
app.py — Clasificacion de emociones caninas en tiempo real.

Optimizaciones vs version anterior:
  - Soporta modelos .tflite (preferidos) y .keras como fallback.
  - Inferencia en hilo separado: la camara nunca se traba esperando al modelo.
  - Con TFLite, el interprete es ~5-10x mas rapido que TF completo en CPU.

Uso:
    python app.py

Primero converte el modelo con convert_to_tflite.py para maxima velocidad.
Presiona 'q' o Esc para salir.
"""

import os
import sys
import glob
import json
import threading
import time

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Constantes ─────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
HISTORIES_DIR = os.path.join(os.path.dirname(__file__), "histories")
CLASSES       = ["angry", "happy", "relaxed", "sad"]
CONF_THRESH   = 0.45

CLASS_COLORS = {
    "angry":   (0,   50,  220),
    "happy":   (0,  200,   50),
    "relaxed": (200, 150,   0),
    "sad":     (180,  80, 180),
}

# Cada backbone espera un tamaño y un preprocesado de imagen distintos.
# InceptionV3 (el modelo original) necesita [-1,1] a mano; los backbones nuevos
# (mobilenetv3, efficientnetv2b0, entrenados con training/train_lightweight.py)
# ya traen el rescaling adentro del modelo y esperan pixeles crudos en [0,255].
BACKBONE_PREPROCESS = {
    "inceptionv3": lambda img: (img.astype("float32") / 127.5) - 1.0,
    "default":     lambda img: img.astype("float32"),
}

# En el set de 1000 clases estandar de ImageNet, los indices 151 a 268 son
# exactamente las 118 razas de perro (de "Chihuahua" a "Mexican_hairless" --
# verificado contra imagenet_class_index.json). Sirve para el filtro de
# "¿hay un perro?" sin tener que entrenar nada nuevo.
DOG_INDEX_START = 151
DOG_INDEX_END = 268  # inclusive
DOG_GATE_THRESH = 0.3  # calibrado: fotos reales de perro dan 0.72-0.95, ruido da ~0.03


def resolve_model_config(model_path: str):
    """
    Determina (img_size, preprocess_fn, backbone) para un modelo.

    Los modelos de training/train_lightweight.py guardan un
    histories/<nombre>_metadata.json con esta info -- es la fuente confiable.
    Si no existe (p. ej. un modelo del notebook original, de antes de que
    existiera este metadata), se asume InceptionV3 por compatibilidad hacia atras.
    """
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    meta_path = os.path.join(HISTORIES_DIR, f"{base_name}_metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        backbone = meta.get("backbone", "default")
        img_size = tuple(meta.get("img_size", (224, 224)))
        preprocess_fn = BACKBONE_PREPROCESS.get(backbone, BACKBONE_PREPROCESS["default"])
        return img_size, preprocess_fn, backbone

    return (299, 299), BACKBONE_PREPROCESS["inceptionv3"], "inceptionv3 (sin metadata, asumido)"


# ── Buscar modelos ──────────────────────────────────────────────────────────────
def list_models() -> list[tuple[str, str]]:
    """
    Devuelve lista de (path, tipo) donde tipo es 'tflite' o 'keras'.
    Prioriza .tflite sobre .keras con el mismo nombre base.
    """
    tflites = {os.path.basename(p).replace(".tflite", ""): p
               for p in sorted(glob.glob(os.path.join(MODELS_DIR, "*.tflite")))}
    keras_m = {os.path.basename(p).replace(".keras", ""): p
               for p in sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))}

    results = []
    seen = set()

    # Primero los que tienen .tflite (rapidos)
    for name, path in sorted(tflites.items()):
        results.append((path, "tflite"))
        seen.add(name)

    # Despues los .keras que no tienen .tflite todavia
    for name, path in sorted(keras_m.items()):
        if name not in seen:
            results.append((path, "keras"))

    return results


def choose_model(models: list[tuple[str, str]]) -> tuple[str, str]:
    print("\n+------------------------------------------------------+")
    print("|            Clasificador de Emociones Caninas          |")
    print("+------------------------------------------------------+\n")

    if not models:
        print(f"[ERROR] No se encontraron modelos en: {MODELS_DIR}")
        print("        Corre el notebook para generar el modelo.")
        sys.exit(1)

    print("Modelos disponibles:")
    for i, (path, tipo) in enumerate(models, 1):
        name    = os.path.basename(path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        tag     = "[RAPIDO - TFLite]" if tipo == "tflite" else "[keras - converte con convert_to_tflite.py]"
        print(f"  [{i}] {name}  ({size_mb:.1f} MB)  {tag}")

    print()
    while True:
        raw = input(f"Elegi un modelo [1-{len(models)}]: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        print(f"    Ingresa un numero entre 1 y {len(models)}.")


# ── Preprocesamiento (compartido por ambos backends) ───────────────────────────
def preprocess(frame: np.ndarray, img_size, preprocess_fn) -> np.ndarray:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size, interpolation=cv2.INTER_LINEAR)
    tensor  = preprocess_fn(resized)
    return np.expand_dims(tensor, axis=0)


# ── Backend TFLite (liviano, rapido) ───────────────────────────────────────────
class TFLitePredictor:
    def __init__(self, path: str, img_size, preprocess_fn):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_idx  = self.interpreter.get_input_details()[0]["index"]
        self.output_idx = self.interpreter.get_output_details()[0]["index"]
        self.img_size = img_size
        self.preprocess_fn = preprocess_fn

    def predict_frame(self, frame: np.ndarray) -> np.ndarray:
        tensor = preprocess(frame, self.img_size, self.preprocess_fn)
        self.interpreter.set_tensor(self.input_idx, tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_idx)[0]


# ── Backend Keras (completo, mas pesado) ───────────────────────────────────────
class KerasPredictor:
    def __init__(self, path: str, img_size, preprocess_fn):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        self.img_size = img_size
        self.preprocess_fn = preprocess_fn
        # Calentamiento
        self.model.predict(np.zeros((1, *img_size, 3), dtype="float32"), verbose=0)

    def predict_frame(self, frame: np.ndarray) -> np.ndarray:
        tensor = preprocess(frame, self.img_size, self.preprocess_fn)
        return self.model.predict(tensor, verbose=0)[0]


# ── Filtro de "¿hay un perro?" ──────────────────────────────────────────────────
class DogGate:
    """
    Filtro liviano que corre ANTES del clasificador de emociones. Usa
    MobileNetV3Small pre-entrenado en ImageNet, SIN fine-tuning -- no hace falta
    entrenar nada, el modelo ya sabe reconocer razas de perro de fabrica.

    Suma la probabilidad de las 118 razas de perro (indices 151-268) en vez de
    mirar solo la clase top-1: la confianza suele repartirse entre varias razas
    parecidas sin que ninguna domine, y sumarlas da una señal mucho mas estable
    de "esto es un perro" que cualquiera de ellas por separado.
    """
    IMG_SIZE = (224, 224)

    def __init__(self):
        import tensorflow as tf
        self.model = tf.keras.applications.MobileNetV3Small(weights="imagenet", include_top=True)

    def is_dog(self, frame: np.ndarray) -> tuple[bool, float]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        tensor = np.expand_dims(resized.astype("float32"), axis=0)  # [0,255] crudo, el modelo rescala adentro
        probs = self.model.predict(tensor, verbose=0)[0]
        dog_prob = float(probs[DOG_INDEX_START:DOG_INDEX_END + 1].sum())
        return dog_prob >= DOG_GATE_THRESH, dog_prob


# ── Hilo de inferencia ─────────────────────────────────────────────────────────
class InferenceThread(threading.Thread):
    """
    Corre la inferencia en background.
    El hilo principal solo lee los resultados cuando estan listos,
    sin bloquearse esperando al modelo.
    """
    def __init__(self, predictor, dog_gate: "DogGate | None" = None):
        super().__init__(daemon=True)
        self.predictor   = predictor
        self.dog_gate    = dog_gate
        self._lock       = threading.Lock()
        self._frame      = None          # frame pendiente de procesar
        self._new_frame  = threading.Event()
        self._result     = ("?", 0.0, True)   # (label, conf, hay_perro)
        self._running    = True

    def submit(self, frame: np.ndarray):
        """Envia un nuevo frame para clasificar (non-blocking)."""
        with self._lock:
            self._frame = frame.copy()
        self._new_frame.set()

    def get_result(self) -> tuple[str, float, bool]:
        with self._lock:
            return self._result

    def stop(self):
        self._running = False
        self._new_frame.set()

    def run(self):
        while self._running:
            self._new_frame.wait()
            self._new_frame.clear()
            if not self._running:
                break

            with self._lock:
                frame = self._frame

            if frame is None:
                continue

            try:
                hay_perro = True
                if self.dog_gate is not None:
                    hay_perro, _ = self.dog_gate.is_dog(frame)

                if hay_perro:
                    probs = self.predictor.predict_frame(frame)
                    idx   = int(np.argmax(probs))
                    label = CLASSES[idx]
                    conf  = float(probs[idx])
                else:
                    label, conf = "?", 0.0

                with self._lock:
                    self._result = (label, conf, hay_perro)
            except Exception as e:
                print(f"[WARN] Error en inferencia: {e}")


# ── Overlay ────────────────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, label: str, confidence: float,
                 fps: float, backend: str, hay_perro: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if not hay_perro:
        color = (90, 90, 90)
        text  = "No se detecta un perro"
        confidence = 0.0
    elif confidence >= CONF_THRESH:
        color = CLASS_COLORS.get(label, (255, 255, 255))
        text  = f"{label.upper()}  {confidence*100:.1f}%"
    else:
        color = (80, 80, 80)
        text  = f"Buscando...  {confidence*100:.1f}%"

    bar_h = 72
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), color, -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Texto principal
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 1.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = h - bar_h + th + 14

    cv2.putText(frame, text, (tx+2, ty+2), font, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,   ty  ), font, scale, (255,255,255), thick, cv2.LINE_AA)

    # Barra de confianza
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (0, h-bar_h-5), (bar_w, h-bar_h), color, -1)

    # Info HUD (arriba izquierda)
    hud = f"FPS: {fps:.1f}  |  {backend}  |  'q' para salir"
    cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ── Bucle principal ─────────────────────────────────────────────────────────────
def run(model_path: str, tipo: str) -> None:
    print(f"\nCargando modelo ({tipo}): {os.path.basename(model_path)} ...")

    img_size, preprocess_fn, backbone = resolve_model_config(model_path)
    print(f"  Config detectada: backbone={backbone}  img_size={img_size}")

    if tipo == "tflite":
        predictor  = TFLitePredictor(model_path, img_size, preprocess_fn)
        backend_id = "TFLite"
    else:
        print("  (esto puede tardar un momento la primera vez)")
        predictor  = KerasPredictor(model_path, img_size, preprocess_fn)
        backend_id = "Keras"

    print("Modelo listo.")

    print("Cargando filtro de detección de perro (MobileNetV3Small, ImageNet)...")
    dog_gate = DogGate()
    print("Filtro listo.")

    # Arrancar hilo de inferencia
    worker = InferenceThread(predictor, dog_gate)
    worker.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la camara (indice 0).")
        worker.stop()
        sys.exit(1)

    # Resolucion mas baja = mas FPS en la camara, suficiente para inferencia
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Dog Emotion Classifier - TP Final IIA"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[INFO] Camara activa. Mostra un perro frente a la camara.")
    print("       Presiona 'q' o Esc para salir.\n")

    SUBMIT_EVERY = 5      # enviar al hilo de inferencia cada N frames
    frame_count  = 0
    t_prev       = time.perf_counter()
    fps          = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Calcular FPS de display
        t_now = time.perf_counter()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        # Enviar frame al hilo de inferencia (non-blocking)
        if frame_count % SUBMIT_EVERY == 0:
            worker.submit(frame)

        label, confidence, hay_perro = worker.get_result()
        display = draw_overlay(frame, label, confidence, fps, backend_id, hay_perro)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nApp cerrada. Hasta luego!")


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    models = list_models()
    path, tipo = choose_model(models)
    run(path, tipo)
