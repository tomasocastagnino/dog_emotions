// Dog Emotion Web -- inferencia 100% en el navegador con TensorFlow.js.
// El orden de CLASSES tiene que coincidir con el orden alfabetico de carpetas
// que usa training/train_lightweight.py (angry=0, happy=1, relaxed=2, sad=3),
// que es el orden que aprendio el modelo durante el entrenamiento.
const CLASSES = ["angry", "happy", "relaxed", "sad"];
const IMG_SIZE = 224;
const CONF_THRESH = 0.45;
const INFER_INTERVAL_MS = 400; // ~2.5 predicciones/seg, de sobra para que se sienta fluido

// Filtro de "¿hay un perro?": MobileNetV3Small pre-entrenado en ImageNet, sin
// fine-tuning. En el set de 1000 clases estandar de ImageNet, los indices 151 a
// 268 son las 118 razas de perro (de "Chihuahua" a "Mexican_hairless"). Sumar
// la probabilidad de todas esas clases da una señal de "es un perro" mas
// estable que mirar solo la clase top-1 (mismo criterio que usa app.py).
const DOG_INDEX_START = 151;
const DOG_INDEX_END = 268; // inclusive
const DOG_GATE_THRESH = 0.3; // calibrado: fotos reales de perro dan 0.72-0.95, ruido da ~0.03

const CLASS_INFO = {
  angry:   { label: "Enojado 😠", color: "#dc3220" },
  happy:   { label: "Feliz 😄",   color: "#2fa63f" },
  relaxed: { label: "Relajado 😌", color: "#c69214" },
  sad:     { label: "Triste 😢",  color: "#8148b3" },
};

const video      = document.getElementById("video");
const startBtn   = document.getElementById("startBtn");
const statusEl   = document.getElementById("status");
const resultEl   = document.getElementById("result");
const labelEl    = document.getElementById("label");
const confBarEl  = document.getElementById("confBar");
const confTextEl = document.getElementById("confText");

let model = null;
let gateModel = null;

async function loadModel() {
  statusEl.textContent = "Cargando modelos (unos MB, una sola vez)...";
  [model, gateModel] = await Promise.all([
    tf.loadGraphModel("model/model.json"),
    tf.loadGraphModel("model_gate/model.json"),
  ]);
  // Calentamiento: la primera inferencia real siempre es mas lenta.
  tf.tidy(() => {
    model.predict(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]));
    gateModel.predict(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]));
  });
  statusEl.textContent = "";
}

async function startCamera() {
  startBtn.disabled = true;
  statusEl.textContent = "Pidiendo acceso a la cámara...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    startBtn.hidden = true;
    resultEl.hidden = false;

    if (!model) await loadModel();

    requestAnimationFrame(loop);
  } catch (err) {
    statusEl.textContent = "No se pudo acceder a la cámara: " + err.message;
    startBtn.disabled = false;
  }
}

function predictFrame() {
  if (!model || !gateModel || video.readyState < 2) return;
  tf.tidy(() => {
    const input = tf.browser
      .fromPixels(video)
      .resizeBilinear([IMG_SIZE, IMG_SIZE])
      .toFloat()
      .expandDims(0); // (1, 224, 224, 3), en [0,255] crudo -- los modelos rescalan adentro

    // Test-time augmentation: promediar la prediccion del frame tal cual y de
    // su espejado horizontal. Los dos modelos vieron flips durante su propio
    // entrenamiento (RandomFlip / augmentation de ImageNet), asi que consultar
    // ambas vistas y promediar suaviza un poco el ruido de una sola pasada,
    // sin reentrenar nada -- el costo es una segunda pasada por frame, que
    // estos modelos livianos absorben sin problema.
    const flipped = tf.reverse(input, [2]);

    const gateProbs = gateModel.predict(input).add(gateModel.predict(flipped)).div(2).dataSync();
    let dogProb = 0;
    for (let i = DOG_INDEX_START; i <= DOG_INDEX_END; i++) dogProb += gateProbs[i];

    if (dogProb < DOG_GATE_THRESH) {
      renderNoDog();
      return;
    }

    const probs = model.predict(input).add(model.predict(flipped)).div(2).dataSync();
    renderResult(probs);
  });
}

function renderNoDog() {
  labelEl.textContent = "No se detecta un perro";
  confBarEl.style.background = "#888";
  confBarEl.style.width = "0%";
  confTextEl.textContent = "";
}

function renderResult(probs) {
  let idx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[idx]) idx = i;
  }
  const conf = probs[idx];
  const cls = CLASSES[idx];
  const pct = Math.round(conf * 100);

  if (conf < CONF_THRESH) {
    labelEl.textContent = "Buscando...";
    confBarEl.style.background = "#888";
  } else {
    const info = CLASS_INFO[cls];
    labelEl.textContent = info.label;
    confBarEl.style.background = info.color;
  }
  confBarEl.style.width = pct + "%";
  confTextEl.textContent = pct + "%";
}

let lastInferTime = 0;
function loop(timestamp) {
  if (timestamp - lastInferTime > INFER_INTERVAL_MS) {
    lastInferTime = timestamp;
    predictFrame();
  }
  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", startCamera);
