# Dog Emotion Web

Clasificador de emociones caninas (`angry`, `happy`, `relaxed`, `sad`) pensado para
correr como página web: activás la cámara del celular o de la compu y el modelo
predice la emoción del perro **corriendo 100% en el navegador**, sin mandar video a
ningún servidor.

**🔴 Probala en vivo: <https://tomasocastagnino.github.io/dog_emotions/>**

## Origen del proyecto

La idea nace de un trabajo práctico de la materia *Introducción a la Inteligencia
Artificial* (UNR), donde entrenamos un clasificador con InceptionV3 (76% accuracy,
124 MB, corriendo en una app de escritorio con la cámara). El repositorio real de
ese trabajo es [gaspigz/TP-FINAL-IIA](https://github.com/gaspigz/TP-FINAL-IIA) —
crédito de origen ahí. Este repo **no es ese trabajo**: es una evolución
independiente, con objetivos propios — mejor accuracy, un modelo mucho más
liviano, y despliegue real como sitio web (GitHub Pages + TensorFlow.js) en vez
de una app de escritorio local.

## Estado actual

- [x] Modelo original de referencia (InceptionV3, 76% acc, 124 MB) — ver [gaspigz/TP-FINAL-IIA](https://github.com/gaspigz/TP-FINAL-IIA).
- [x] Reentrenar con una arquitectura liviana — MobileNetV3Large ganó, 80.75% acc / 24.3 MB (tabla completa abajo).
- [x] Convertir el modelo final a TensorFlow.js (`docs/model/`, 3.1 MB).
- [x] Armar la página (cámara + inferencia en el navegador) — `docs/index.html` + `docs/app.js`.
- [x] Filtro de "¿hay un perro?" antes de mostrar una emoción (`docs/model_gate/`, ver abajo).
- [x] Publicada en GitHub Pages: <https://tomasocastagnino.github.io/dog_emotions/>.

## Modelo final

Se probaron 4 configuraciones sobre el mismo split que el trabajo original (comparable
1 a 1 contra el 0.760 de InceptionV3):

| Corrida | test_acc | tamaño | test_loss |
| --- | --- | --- | --- |
| **mobilenetv3_n30_do20_aug** (elegido) | **0.8075** | **24.3 MB** | 0.796 |
| mobilenetv3_n20_do20_aug | 0.7800 | 19.4 MB | 0.748 |
| efficientnetv2b0_n20_do20_aug | 0.7738 | 29.8 MB | 0.588 |
| mobilenetv3_n10_do30_aug | 0.7600 | 16.3 MB | 0.631 |

Ganó `mobilenetv3_n30_do20_aug` (MobileNetV3Large, 30 capas del backbone
descongeladas, dropout 0.2, augmentation): mejor accuracy de las 4, y sigue siendo
~5x más chico que InceptionV3. Convertido a TensorFlow.js queda en 3.1 MB
(`docs/model/`) — el mismo modelo que sirve `docs/index.html`.

Análisis completo (matriz de confusión, curvas de entrenamiento, comparación
por clase contra el modelo original) en **[REPORT.md](REPORT.md)**.

Después se probó tres veces sumar datos extra (Flickr filtrado por el propio
modelo, y otro dataset de Kaggle) para intentar superar este resultado —
**ninguno lo superó**, y una revisión manual de las confusiones más comunes
encontró la causa: la etiqueta `sad` del dataset original mezcla dos señales
distintas (cara/postura triste vs. contexto de refugio/jaula), algo que
ningún volumen de datos extra puede arreglar por sí solo. Los tres intentos,
los números por clase y las fotos concretas que se revisaron están en
[REPORT.md](REPORT.md).

### Filtro de "¿hay un perro?"

El clasificador de emociones siempre devuelve una de las 4 clases, incluso
apuntando a algo que no es un perro. Para evitarlo, antes de clasificar la
emoción se corre un filtro con `MobileNetV3Small` pre-entrenado en ImageNet
**sin ningún fine-tuning** — no hizo falta entrenar nada: en el set de 1000
clases estándar de ImageNet, los índices 151 a 268 son exactamente las 118
razas de perro, y sumar su probabilidad combinada da una señal muy confiable
de "esto es un perro" (calibrado con fotos reales del dataset: 0.72–0.95, contra
~0.03–0.04 en ruido aleatorio). Pesa 2,6 MB (`docs/model_gate/`) y corre igual
en la página web y en `app.py`.

## Estructura del repo

```
.
├── training/
│   ├── train_lightweight.py       # entrena MobileNetV3Large / EfficientNetV2B0
│   ├── convert_to_tfjs.py         # convierte el .keras ganador a docs/model/
│   ├── convert_dog_gate.py        # genera docs/model_gate/ (filtro "¿hay un perro?")
│   ├── filter_extra_images.py     # cura data/images/ usando el modelo actual como filtro
│   ├── build_archive_manifest.py  # arma un manifest desde archive/ (otro dataset de Kaggle)
│   └── generate_report_assets.py  # genera los gráficos de REPORT.md
├── app.py                       # app de escritorio (prototipo original, cámara local)
├── convert_to_tflite.py         # conversión .keras -> .tflite para la app de escritorio
├── docs/                        # la página web -> GitHub Pages (ya publicada, ver arriba)
│   ├── index.html, style.css, app.js
│   ├── model/                   # modelo de emociones en TensorFlow.js (3.1 MB)
│   └── model_gate/              # filtro "¿hay un perro?" en TensorFlow.js (2.6 MB)
├── models/, histories/          # modelos entrenados + su historial (no se versionan)
├── data/                        # datasets locales (no se versionan, ver más abajo)
├── report_assets/               # gráficos de REPORT.md (comparación, curvas, matriz de confusión)
├── REPORT.md                    # análisis completo de resultados
├── LICENSE                      # MIT
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
  test_acc, tamaño en MB y segundos que tardó. Sirve para no perder de vista qué
  configuración exacta generó cada modelo (antes había que reconstruirlo a mano
  revisando el notebook).

Parámetros disponibles: `--n-capas` (capas del backbone a descongelar, default 20),
`--dropout` (default 0.2), `--augment`/`--no-augment`, `--epochs`, `--patience`.
Correr `python training/train_lightweight.py --help` para el detalle.

**Dónde correrlo:** para este dataset (3200 imágenes de train+val, arquitecturas
más chicas que InceptionV3) alcanza con una notebook común — probalo primero en tu
Mac. Si notás que tarda mucho, pasate a Google Colab (GPU T4 gratis, no necesita
setup adicional más que subir el script o clonar el repo). Sumar datos extra
(ver abajo) hace las corridas bastante más lentas — ahí sí una GPU dedicada
(por ej. una RTX 3060 Ti) empieza a rendir.

### Sumar datos extra (experimental)

```bash
python training/filter_extra_images.py --model models/mobilenetv3_n30_do20_aug.keras
python training/train_lightweight.py --backbone mobilenetv3 --n-capas 30 --dropout 0.2 \
  --extra-manifest data/images_filtered_manifest.csv --extra-classes relaxed,sad --extra-weight 0.5
```

`--extra-manifest` suma imágenes extra solo a `train` (nunca a val/test), con
`sample_weight` para pesar menos que el dataset curado. `filter_extra_images.py`
cura `data/images/` (recolectadas de Flickr, sin revisar) usando el modelo
actual como filtro; `build_archive_manifest.py` hace lo mismo para cualquier
otro dataset de Kaggle que agregues en `archive/`, excluyendo automáticamente
lo que ya esté en el test set. **Se probó tres veces y ninguna superó al
modelo sin datos extra** — el detalle completo, incluyendo por qué, está en
[REPORT.md](REPORT.md).

## Datos

El dataset de entrenamiento ([Dog Emotion, Kaggle](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion))
se descarga solo con `kagglehub` — no hace falta tenerlo local.

`data/` y `archive/` (si existen en tu copia local) tienen datasets extra para
explorar/experimentar y **no se versionan** (están en `.gitignore`) — ninguno
tiene licencia confirmada para redistribuir, así que no deberían terminar en
el repo de todas formas:
- `data/images/`: ~15.900 imágenes recolectadas de Flickr, sin balancear entre
  clases y sin curar. Se probó sumarlas al entrenamiento filtradas por el
  propio modelo (`training/filter_extra_images.py` genera
  `data/images_filtered_manifest.csv`) — no mejoró el resultado, ver
  [REPORT.md](REPORT.md).
- `archive/`: otro dataset de Kaggle ("Final dog dataset") que se probó de la
  misma forma (`training/build_archive_manifest.py` genera
  `data/archive_manifest.csv`, excluyendo lo que ya está en el test set).
  Tampoco mejoró el resultado.

## Página web

`docs/` tiene la página lista: `index.html` + `style.css` + `app.js` + `model/`
(el modelo ya convertido a TensorFlow.js, 3.1 MB). Todo corre en el navegador —
la cámara nunca manda video a ningún lado.

**Probarla en local** (la cámara necesita HTTPS o `localhost`, por eso no alcanza
con abrir el archivo directo):

```bash
python3 -m http.server 8080 --directory docs
# abrir http://localhost:8080
```

Publicada en GitHub Pages, se actualiza sola con cada push a `main` que toque
`docs/`.

**Si reentrenás un modelo nuevo y lo dejás mejor que el actual**, así se
regeneran los modelos de `docs/`:

```bash
python training/convert_to_tfjs.py --model models/<tu-modelo-nuevo>.keras   # docs/model/
python training/convert_dog_gate.py                                        # docs/model_gate/
```

## App de escritorio (prototipo original)

`app.py` y `convert_to_tflite.py` siguen funcionando para probar cualquier modelo
`.keras`/`.tflite` de `models/` contra la cámara de la compu (ver comentarios en cada
archivo). El foco del proyecto ahora es la versión web, pero esto se mantiene andando
como referencia y para probar modelos rápido sin esperar al armado de la página.

```bash
python app.py
```

## Licencia

[MIT](LICENSE) — el código de este repo, no el dataset (ver "Datos" arriba
para las condiciones de cada uno).
