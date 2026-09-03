#!/usr/bin/env python3
"""RCTL 실행 전에 Colab Python/TensorFlow/GPU runtime을 짧게 확인한다."""

from __future__ import annotations

import json
import platform
import subprocess

import numpy as np
import tensorflow as tf


def main() -> int:
    try:
        gpu_names = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu_names = []
    payload = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", None),
        "tensorflow_gpu_devices": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
        "nvidia_smi_gpus": gpu_names,
        "is_t4": any("T4" in name for name in gpu_names),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["is_t4"] and payload["tensorflow_gpu_devices"] else 2


if __name__ == "__main__":
    exit_status = main()
    if exit_status:
        raise RuntimeError("요청한 T4 또는 TensorFlow GPU device를 확인하지 못했습니다.")
