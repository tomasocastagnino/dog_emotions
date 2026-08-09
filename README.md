# Dog Emotion Web

Clasificador de emociones caninas (`angry`, `happy`, `relaxed`, `sad`) pensado para
correr como página web: activás la cámara del celular o de la compu y el modelo
predice la emoción del perro **corriendo 100% en el navegador**, sin mandar video a
ningún servidor.

## Origen del proyecto

La idea nace de un trabajo práctico de la materia *Introducción a la Inteligencia
Artificial* (UNR), donde entrenamos un clasificador con InceptionV3 (76% accuracy,
124 MB, corriendo en una app de escritorio con la cámara). Ese trabajo queda
documentado en [`original-tp/`](original-tp/) como crédito de origen, pero este repo
**no es ese trabajo**: es una evolución independiente, con objetivos propios —
mejor accuracy, un modelo mucho más liviano, y despliegue real como sitio web
(GitHub Pages + TensorFlow.js) en vez de una app de escritorio local.

## Estado actual

- [x] Modelo original de referencia (InceptionV3, 76% acc, 124 MB) — ver `original-tp/`.
- [ ] Reentrenar con una arquitectura liviana (MobileNetV3Large / EfficientNetV2B0).
- [ ] Convertir el modelo final a TensorFlow.js.
- [ ] Armar la página (cámara + inferencia en el navegador).
- [ ] Publicar en GitHub Pages (sirviendo desde `docs/`).

## Estructura del repo

```
.
├── training/
│   └── train_lightweight.py     # entrena MobileNetV3Large / EfficientNetV2B0
├── original-tp/
│   └── TP_Final_IIA_....ipynb   # notebook del trabajo práctico original (InceptionV3)
├── app.py                       # app de escritorio (prototipo original, cámara local)
├── convert_to_tflite.py         # conversión .keras -> .tflite para la app de escritorio
├── docs/                        # sitio estático -> GitHub Pages (Settings > Pages > /docs)
├── models/, histories/          # modelos entrenados + su historial (no se versionan)
├── data/                        # datasets locales (no se versionan, ver más abajo)
├── requirements.txt
└── README.md
```

## Setup

### 1. Requisito: Python 3.12

TensorFlow todavía **no** publica wheels para Python 3.13/3.14. Si usás una versión
más nueva, `pip install tensorflow` falla con *"No matching distribution found"*.
Usá **Python 3.12**.

En Mac con Homebrew:

```bash
brew install python@3.12
```

En Windows (PowerShell):

```powershell
winget install Python.Python.3.12
```

(o bajalo de [python.org](https://www.python.org/downloads/)).

### 2. Crear y activar el entorno virtual

**Mac / Linux:**

```bash
# parado en la carpeta del proyecto
python3.12 -m venv .venv
source .venv/bin/activate
python --version                 # debe decir 3.12.x
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

> Si PowerShell bloquea el script de activación con un error de *"execution
> policy"*, corré una sola vez:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` y volvé a
> activar.

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Credenciales de Kaggle

El dataset se baja con `kagglehub`, que necesita un **API token** de tu cuenta:

1. Kaggle → tu avatar → **Settings** → sección **API** → **Create New Token**.
2. Se descarga un `kaggle.json` con `{ "username": "...", "key": "..." }`.
3. Poné ese archivo en la **raíz del proyecto** (al lado de `requirements.txt`).

```bash
mv ~/Downloads/kaggle.json ./
chmod 600 kaggle.json     # Mac/Linux: solo tu usuario puede leerlo
```

> `kaggle.json` está en `.gitignore` — nunca se sube al repo. En **Google Colab**
> no hace falta este archivo: `training/train_lightweight.py` detecta el entorno
> solo y usa los Secrets del notebook (`KAGGLE_USERNAME` / `KAGGLE_KEY`) en su lugar.

### 5. GPU local con WSL2 (Windows + NVIDIA)

TensorFlow dejó de soportar GPU en Windows nativo desde la versión 2.11 — en
Windows nativo siempre corre en CPU, aunque tengas CUDA instalado. Para usar una
GPU NVIDIA hace falta WSL2 (Ubuntu dentro de Windows):

```powershell
wsl --install -d Ubuntu
wsl -d Ubuntu -- nvidia-smi      # debe listar tu GPU
```

Dentro de WSL, con [`uv`](https://docs.astral.sh/uv/) para tener Python 3.12 sin
tocar el sistema:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv --python 3.12 ~/venvs/dog-emotion
uv pip install --python ~/venvs/dog-emotion/bin/python \
  "tensorflow[and-cuda]==2.21.0" -r requirements.txt
```

El `[and-cuda]` baja las librerías CUDA/cuDNN como paquetes pip, no hace falta
instalar CUDA aparte. Para CPU nomás (sin GPU), el setup normal de arriba alcanza:
para este dataset (4000 imágenes) hasta CPU pura entrena en minutos.

## Entrenar el modelo liviano

```bash
# correr siempre desde la raíz del repo, no desde adentro de training/
python training/train_lightweight.py --backbone mobilenetv3
python training/train_lightweight.py --backbone efficientnetv2b0
```

Descarga el dataset solo (mismas credenciales que arriba), usa el **mismo split
60/20/20 y la misma seed** que el trabajo original, así el `test_acc` que da es
comparable 1 a 1 contra el 0.760 de InceptionV3. Guarda tres cosas por corrida en
`models/` y `histories/`:

- `models/<nombre-corrida>.keras` — el modelo.
- `histories/<nombre-corrida>_history.pkl` — curvas de loss/accuracy por época.
- `histories/<nombre-corrida>_metadata.json` — arquitectura, hiperparámetros,
  test_acc, tamaño en MB y segundos que tardó. Esto es a propósito: la vez pasada
  perdimos el rastro de qué configuración exacta era `mejor_estrategia_2.keras` y
  hubo que reconstruirlo a mano desde el notebook — con el `.json` al lado de cada
  modelo no vuelve a pasar.

Parámetros disponibles: `--n-capas` (capas del backbone a descongelar, default 20),
`--dropout` (default 0.2), `--augment`/`--no-augment`, `--epochs`, `--patience`.
Correr `python training/train_lightweight.py --help` para el detalle.

**Dónde correrlo:** para este dataset (3200 imágenes de train+val, arquitecturas
más chicas que InceptionV3) alcanza con una notebook común — probalo primero en tu
Mac. Si notás que tarda mucho, pasate a Google Colab (GPU T4 gratis, no necesita
setup adicional más que subir el script o clonar el repo). Una GPU dedicada (por
ej. una RTX 3060 Ti) queda en reserva para más adelante, si hace falta un barrido
más grande de configuraciones o sumar el dataset extra de `data/images/`.

## Datos

El dataset de entrenamiento ([Dog Emotion, Kaggle](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion))
se descarga solo con `kagglehub` — no hace falta tenerlo local.

`data/` (si existe en tu copia local) tiene datasets para mirar/explorar a mano y
**no se versiona** (está en `.gitignore`):
- `data/Dog Emotion/`: copia local del dataset de Kaggle.
- `data/images/`: imágenes adicionales recolectadas para una posible ampliación del
  dataset. Todavía no se usan en el entrenamiento. Antes de sumarlas: no están
  balanceadas entre clases, y su procedencia (nombres de archivo tipo Flickr) sugiere
  que no pasaron por una curación tan cuidada como el dataset de Kaggle — conviene
  revisar una muestra a mano antes de confiar en el set completo, y de todas formas
  no deberían terminar versionadas en un repo público sin confirmar la licencia de
  cada imagen.

## App de escritorio (prototipo original)

`app.py` y `convert_to_tflite.py` siguen funcionando para probar cualquier modelo
`.keras`/`.tflite` de `models/` contra la cámara de la compu (ver comentarios en cada
archivo). El foco del proyecto ahora es la versión web, pero esto se mantiene andando
como referencia y para probar modelos rápido sin esperar al armado de la página.

```bash
python app.py
```
# dog_emotions
