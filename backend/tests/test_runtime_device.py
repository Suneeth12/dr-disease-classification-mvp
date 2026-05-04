from __future__ import annotations

from retina_api.ml.runtime_device import configure_tensorflow_runtime


class FakeDevice:
    def __init__(self, name: str, device_type: str) -> None:
        self.name = name
        self.device_type = device_type


class FakeMixedPrecision:
    def __init__(self) -> None:
        self.policy_name = "float32"

    def set_global_policy(self, policy_name: str) -> None:
        self.policy_name = policy_name

    def global_policy(self):
        return type("Policy", (), {"name": self.policy_name})()


class FakeConfig:
    def __init__(self) -> None:
        self.gpus = [
            FakeDevice("/physical_device:GPU:0", "GPU"),
            FakeDevice("/physical_device:GPU:1", "GPU"),
        ]
        self.visible_devices = None
        self.memory_growth_calls = []

    def list_physical_devices(self, device_type: str):
        return self.gpus if device_type == "GPU" else [FakeDevice("/physical_device:CPU:0", "CPU")]

    def list_logical_devices(self, device_type: str):
        if device_type != "GPU":
            return [FakeDevice("/device:CPU:0", "CPU")]
        if self.visible_devices == []:
            return []
        visible = self.visible_devices if self.visible_devices is not None else self.gpus
        return [FakeDevice(f"/device:GPU:{index}", "GPU") for index, _ in enumerate(visible)]

    def set_visible_devices(self, devices, device_type: str) -> None:
        assert device_type == "GPU"
        self.visible_devices = devices


class FakeExperimentalConfig:
    def __init__(self, config: FakeConfig) -> None:
        self.config = config

    def set_memory_growth(self, device, enabled: bool) -> None:
        self.config.memory_growth_calls.append((device.name, enabled))


class FakeTF:
    def __init__(self) -> None:
        self.config = FakeConfig()
        self.config.experimental = FakeExperimentalConfig(self.config)
        self.keras = type(
            "Keras",
            (),
            {"mixed_precision": FakeMixedPrecision()},
        )()


def test_configure_tensorflow_runtime_uses_gpu_with_memory_growth_by_default() -> None:
    fake_tf = FakeTF()

    info = configure_tensorflow_runtime(tf_module=fake_tf, environ={})

    assert info.device_name == "/GPU:0"
    assert info.device_type == "GPU"
    assert info.gpu_available is True
    assert info.memory_growth_enabled is True
    assert fake_tf.config.memory_growth_calls == [
        ("/physical_device:GPU:0", True),
        ("/physical_device:GPU:1", True),
    ]


def test_configure_tensorflow_runtime_can_select_one_gpu_and_enable_mixed_precision() -> None:
    fake_tf = FakeTF()

    info = configure_tensorflow_runtime(
        tf_module=fake_tf,
        environ={"DR_GPU_DEVICE_INDEX": "1", "DR_MIXED_PRECISION": "1"},
    )

    assert fake_tf.config.visible_devices == [fake_tf.config.gpus[1]]
    assert fake_tf.config.memory_growth_calls == [("/physical_device:GPU:1", True)]
    assert info.device_name == "/GPU:0"
    assert info.physical_gpus == ["/physical_device:GPU:1"]
    assert info.mixed_precision_policy == "mixed_float16"


def test_configure_tensorflow_runtime_respects_force_cpu() -> None:
    fake_tf = FakeTF()

    info = configure_tensorflow_runtime(
        tf_module=fake_tf,
        environ={"DR_FORCE_CPU": "true"},
    )

    assert fake_tf.config.visible_devices == []
    assert info.device_name == "/CPU:0"
    assert info.device_type == "CPU"
    assert info.gpu_available is False
    assert fake_tf.config.memory_growth_calls == []
