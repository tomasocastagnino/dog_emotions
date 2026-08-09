// Dog Emotion Web -- inferencia 100% en el navegador con TensorFlow.js.
// El orden de CLASSES tiene que coincidir con el orden alfabetico de carpetas
// que usa training/train_lightweight.py (angry=0, happy=1, relaxed=2, sad=3),
// que es el orden que aprendio el modelo durante el entrenamiento.
const CLASSES = ["angry", "happy", "relaxed", "sad"];
const IMG_SIZE = 224;
const CONF_THRESH = 0.45;
const INFER_INTERVAL_MS = 400; // ~2.5 predicciones/seg, de sobra para que se sienta fluido

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

async function loadModel() {
  statusEl.textContent = "Cargando modelo (unos MB, una sola vez)...";
  model = await tf.loadGraphModel("model/model.json");
  // Calentamiento: la primera inferencia real siempre es mas lenta.
  tf.tidy(() => model.predict(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3])));
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
  if (!model || video.readyState < 2) return;
  tf.tidy(() => {
    const input = tf.browser
      .fromPixels(video)
      .resizeBilinear([IMG_SIZE, IMG_SIZE])
      .toFloat()
      .expandDims(0); // (1, 224, 224, 3), en [0,255] crudo -- el modelo rescala adentro
    const probs = model.predict(input).dataSync();
    renderResult(probs);
  });
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
