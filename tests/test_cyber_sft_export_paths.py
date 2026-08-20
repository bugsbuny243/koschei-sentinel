from __future__ import annotations

import pytest

from koschei_sentinel.cyber_sft_export import _validate_adapter_relative_path


@pytest.mark.parametrize(
    "relative",
    (
        "../adapter_model.safetensors",
        "adapter/../adapter_model.safetensors",
        "adapter\\adapter_model.safetensors",
        "adapter_model.safetensors",
        "/adapter/adapter_model.safetensors",
    ),
)
def test_exporter_rejects_non_portable_adapter_paths(relative: str) -> None:
    with pytest.raises(ValueError):
        _validate_adapter_relative_path(relative)


def test_exporter_accepts_trainer_adapter_paths() -> None:
    path = _validate_adapter_relative_path("adapter/adapter_model.safetensors")

    assert path.as_posix() == "adapter/adapter_model.safetensors"
