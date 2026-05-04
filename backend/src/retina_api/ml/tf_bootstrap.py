from __future__ import annotations

import os


def configure_tensorflow_environment() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


configure_tensorflow_environment()
