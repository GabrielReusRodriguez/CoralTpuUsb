#!/usr/bin/env python3
"""Detecta y lista los dispositivos Edge TPU disponibles."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
    except ImportError as exc:
        print("ERROR: pycoral no está instalado.", file=sys.stderr)
        print(f"Detalle: {exc}", file=sys.stderr)
        return 1

    devices = list_edge_tpus()
    if not devices:
        print("No se detectó ningún Edge TPU.")
        print("Comprueba que el Coral USB esté conectado y que el contenedor")
        print("se haya arrancado con --privileged y /dev/bus/usb montado.")
        return 2

    print(f"Edge TPU detectados: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  [{i}] type={device.get('type')} path={device.get('path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
