from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TensorFlowRuntimeInfo:
    device_name: str
    device_type: str
    gpu_available: bool
    physical_gpus: list[str]
    logical_gpus: list[str]
    memory_growth_enabled: bool
    mixed_precision_policy: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _device_names(devices: list[Any]) -> list[str]:
    return [str(getattr(device, "name", device)) for device in devices]


def _load_tensorflow(tf_module: Any | None) -> Any:
    if tf_module is not None:
        return tf_module
    from retina_api.ml import tf_bootstrap  # noqa: F401
    import tensorflow as tf

    return tf


def _current_mixed_precision_policy(tf_module: Any) -> str:
    try:
        return str(tf_module.keras.mixed_precision.global_policy().name)
    except Exception as exc:  # pragma: no cover - defensive around TensorFlow internals
        return f"unknown: {exc}"


def configure_tensorflow_runtime(
    *,
    tf_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> TensorFlowRuntimeInfo:
    tf = _load_tensorflow(tf_module)
    env = environ if environ is not None else os.environ
    notes: list[str] = []

    try:
        physical_gpus = list(tf.config.list_physical_devices("GPU"))
    except Exception as exc:  # pragma: no cover - defensive around TensorFlow internals
        notes.append(f"Failed to list GPU devices: {exc}")
        physical_gpus = []

    force_cpu = _truthy(env.get("DR_FORCE_CPU")) or _truthy(env.get("DR_DISABLE_GPU"))
    if force_cpu:
        if physical_gpus:
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError as exc:
                notes.append(f"Could not hide GPU devices after TensorFlow initialization: {exc}")
        return TensorFlowRuntimeInfo(
            device_name="/CPU:0",
            device_type="CPU",
            gpu_available=False,
            physical_gpus=_device_names(physical_gpus),
            logical_gpus=[],
            memory_growth_enabled=False,
            mixed_precision_policy=_current_mixed_precision_policy(tf),
            notes=notes,
        )

    if not physical_gpus:
        return TensorFlowRuntimeInfo(
            device_name="/CPU:0",
            device_type="CPU",
            gpu_available=False,
            physical_gpus=[],
            logical_gpus=[],
            memory_growth_enabled=False,
            mixed_precision_policy=_current_mixed_precision_policy(tf),
            notes=notes,
        )

    selected_gpus = physical_gpus
    requested_index = env.get("DR_GPU_DEVICE_INDEX")
    if requested_index not in (None, ""):
        try:
            selected_index = int(str(requested_index))
        except ValueError as exc:
            raise ValueError("DR_GPU_DEVICE_INDEX must be an integer") from exc
        if selected_index < 0 or selected_index >= len(physical_gpus):
            raise ValueError(
                f"DR_GPU_DEVICE_INDEX={selected_index} is out of range for {len(physical_gpus)} visible GPU(s)"
            )
        selected_gpus = [physical_gpus[selected_index]]
        tf.config.set_visible_devices(selected_gpus, "GPU")

    memory_growth_enabled = _truthy(env.get("DR_GPU_MEMORY_GROWTH"), default=True)
    if memory_growth_enabled:
        for gpu in selected_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                notes.append(f"Could not enable GPU memory growth for {getattr(gpu, 'name', gpu)}: {exc}")

    if _truthy(env.get("DR_MIXED_PRECISION")):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception as exc:  # pragma: no cover - defensive around TensorFlow internals
            notes.append(f"Could not enable mixed precision: {exc}")

    try:
        logical_gpus = list(tf.config.list_logical_devices("GPU"))
    except Exception as exc:  # pragma: no cover - defensive around TensorFlow internals
        notes.append(f"Failed to list logical GPU devices: {exc}")
        logical_gpus = []

    return TensorFlowRuntimeInfo(
        device_name="/GPU:0",
        device_type="GPU",
        gpu_available=True,
        physical_gpus=_device_names(selected_gpus),
        logical_gpus=_device_names(logical_gpus),
        memory_growth_enabled=memory_growth_enabled,
        mixed_precision_policy=_current_mixed_precision_policy(tf),
        notes=notes,
    )
