# Coral USB + TensorFlow Lite (Docker)

Entorno Docker para ejecutar inferencia ML en un **Google Coral USB Accelerator** (Edge TPU) desde Python, con TensorFlow CPU disponible para desarrollo y preparación de modelos.

## Requisitos del host

- Linux x86_64 (probado conceptualmente sobre Ubuntu 24.04)
- [Docker](https://docs.docker.com/get-docker/) y Docker Compose
- Coral USB Accelerator conectado (preferible puerto **USB 3.0**)

No hace falta instalar el runtime Coral en el host: va dentro de la imagen.

Comprueba que el dispositivo se ve en el host:

```bash
lsusb | grep -iE 'Google|18d1|Global Unichip'
```

## Qué incluye la imagen

| Componente | Uso |
|------------|-----|
| `libedgetpu1-std` | Runtime Edge TPU (frecuencia estándar, más estable/fresco) |
| `python3-pycoral` | API Python + `tflite-runtime` compatible |
| `tensorflow` (CPU) | ML general y preparación de modelos en CPU |
| Pillow, NumPy | Preproceso de imágenes / arrays |

Base: **Debian 11 (bullseye)** + **Python 3.9** (compatible con los paquetes oficiales de Coral).

### TensorFlow vs Edge TPU

El Coral **no ejecuta TensorFlow completo**. Solo acelera modelos **TensorFlow Lite** compilados para Edge TPU (`*_edgetpu.tflite`), vía PyCoral / TFLite.

Usa TensorFlow en CPU para entrenar o convertir; para la TPU usa PyCoral con un `.tflite` ya compilado.

## Uso rápido

```bash
# Construir
docker compose -f docker/docker-compose.yml --project-directory docker build

# Ejecutar un script Python en el contenedor
./run-coral.sh examples/check_tpu.py
./run-coral.sh examples/classify_image.py

# Shell interactivo con acceso USB
docker compose -f docker/docker-compose.yml --project-directory docker run --rm coral
```

Equivalente sin Compose:

```bash
docker build -t coral-tpu docker
docker run --rm -it --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$PWD":/workspace \
  -w /workspace \
  coral-tpu bash
```

`--privileged` y el montaje de `/dev/bus/usb` son necesarios para que `libedgetpu` abra el acelerador.

## Ejemplos

- `examples/check_tpu.py` — lista los Edge TPU visibles.
- `examples/classify_image.py` — descarga (si hace falta) MobileNet V2 birds + imagen de prueba y clasifica en la TPU.

```bash
./run-coral.sh examples/classify_image.py \
  --model ruta/modelo_edgetpu.tflite \
  --labels ruta/labels.txt \
  --input ruta/imagen.jpg
```

## Modelos compatibles

Para correr en la TPU el modelo debe:

1. Estar en formato **TensorFlow Lite** cuantizado (int8).
2. Haberse compilado con el [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/).

Este contenedor se centra en **ejecutar** inferencia. La compilación Edge TPU suele hacerse en otra máquina/etapa (el compilador no forma parte de esta imagen).

Modelos de ejemplo: [coral.ai/models](https://coral.ai/models/).

## Runtime std vs max

Por defecto se instala `libedgetpu1-std` (menos calor). Para máxima frecuencia, en `docker/Dockerfile` sustituye por `libedgetpu1-max` y reconstruye. El dispositivo puede calentarse mucho en modo max.

## Problemas frecuentes

| Síntoma | Qué revisar |
|---------|-------------|
| `Failed to load delegate from libedgetpu.so.1` | Contenedor con `--privileged` y `/dev/bus/usb` montado; Coral enchufado |
| Primera inferencia falla tras arrancar | Reintentar; en Docker a veces falla el primer intento |
| `No se detectó ningún Edge TPU` | `lsusb` en host; cable USB 3.0; reconectar el dispositivo |
| Conflictos de versión PyCoral / TF | Mantén Debian 11 + paquetes apt de Coral; no actualices `tflite-runtime` a mano |

## Estructura

```
.
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── examples/
│   ├── check_tpu.py
│   └── classify_image.py
├── LICENSE
├── run-coral.sh
└── README.md
```

## Licencia

Este proyecto se distribuye bajo la [GNU General Public License v3.0](LICENSE).
