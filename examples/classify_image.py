#!/usr/bin/env python3
"""Clasificación de imagen de ejemplo en el Coral Edge TPU.

Descarga bajo demanda el modelo MobileNet V2 (birds) y una imagen de prueba
oficiales de Coral, luego ejecuta inferencia con PyCoral.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "test_data"

FILES = {
    "model": (
        "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
        "https://github.com/google-coral/test_data/raw/master/"
        "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
    ),
    "labels": (
        "inat_bird_labels.txt",
        "https://github.com/google-coral/test_data/raw/master/inat_bird_labels.txt",
    ),
    "image": (
        "parrot.jpg",
        "https://github.com/google-coral/test_data/raw/master/parrot.jpg",
    ),
}


def ensure_file(name: str, url: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    if path.exists():
        return path
    print(f"Descargando {name}...")
    urllib.request.urlretrieve(url, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Ruta a un modelo *_edgetpu.tflite")
    parser.add_argument("--labels", type=Path, help="Ruta al archivo de etiquetas")
    parser.add_argument("--input", type=Path, help="Ruta a la imagen de entrada")
    parser.add_argument("--top-k", type=int, default=3, help="Número de resultados")
    parser.add_argument("--count", type=int, default=5, help="Repeticiones de inferencia")
    args = parser.parse_args()

    try:
        from pycoral.adapters import classify, common
        from pycoral.utils.dataset import read_label_file
        from pycoral.utils.edgetpu import make_interpreter
    except ImportError as exc:
        print(f"ERROR: no se pudo importar pycoral: {exc}", file=sys.stderr)
        return 1

    model_path = args.model or ensure_file(*FILES["model"])
    labels_path = args.labels or ensure_file(*FILES["labels"])
    image_path = args.input or ensure_file(*FILES["image"])

    labels = read_label_file(str(labels_path))
    interpreter = make_interpreter(str(model_path))
    interpreter.allocate_tensors()

    size = common.input_size(interpreter)
    image = Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)
    common.set_input(interpreter, image)

    print("----INFERENCE TIME----")
    print(
        "Note: The first inference on Edge TPU is slow because it includes "
        "loading the model into Edge TPU memory."
    )
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{elapsed_ms:.1f}ms")

    classes = classify.get_classes(interpreter, top_k=args.top_k)
    print("-------RESULTS--------")
    for c in classes:
        label = labels.get(c.id, c.id)
        print(f"{label}: {c.score:.5f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        # Primera inferencia tras boot en Docker a veces falla; reintentar suele bastar.
        print(f"ERROR al usar el Edge TPU: {exc}", file=sys.stderr)
        print(
            "Si es el primer intento tras arrancar, vuelve a ejecutar el script.",
            file=sys.stderr,
        )
        sys.exit(2)
