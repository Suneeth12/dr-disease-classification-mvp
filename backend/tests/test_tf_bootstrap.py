from __future__ import annotations


def test_configure_tensorflow_environment_sets_quiet_logging_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    from retina_api.ml.tf_bootstrap import configure_tensorflow_environment

    configure_tensorflow_environment()

    assert __import__("os").environ["TF_CPP_MIN_LOG_LEVEL"] == "2"


def test_configure_tensorflow_environment_preserves_existing_log_level(monkeypatch) -> None:
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "0")

    from retina_api.ml.tf_bootstrap import configure_tensorflow_environment

    configure_tensorflow_environment()

    assert __import__("os").environ["TF_CPP_MIN_LOG_LEVEL"] == "0"
