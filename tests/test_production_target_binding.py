from __future__ import annotations

import pytest
from pydantic import ValidationError

from koschei_sentinel.production_target_binding import (
    ProductionTargetBinding,
    require_verified_production_target_binding,
)

_ZERO = "0" * 64
_ONE = "1" * 64
_TWO = "2" * 64
_THREE = "3" * 64
_FOUR = "4" * 64


def _binding(**overrides):
    payload = {
        "model_ref": "sentinel-397b-35b",
        "model_revision": _FOUR,
        "architecture_manifest_sha256": _ONE,
        "router_manifest_sha256": _TWO,
        "expert_topology_manifest_sha256": _THREE,
        "trainer_adapter_version": "4.5.2",
        "production_training_allowed": False,
        "verification_status": "unverified",
        "blockers": ["not yet verified"],
    }
    payload.update(overrides)
    return ProductionTargetBinding(**payload)


def test_unverified_binding_is_blocked() -> None:
    binding = _binding()
    with pytest.raises(ValueError, match="not verified"):
        require_verified_production_target_binding(binding)


def test_cannot_enable_training_without_verified_status() -> None:
    with pytest.raises(ValidationError):
        _binding(production_training_allowed=True)


def test_verified_binding_cannot_keep_blockers() -> None:
    with pytest.raises(ValidationError):
        _binding(
            verification_status="verified",
            production_training_allowed=True,
            blockers=["stale blocker"],
        )


def test_verified_binding_can_authorize_production_training() -> None:
    binding = _binding(
        verification_status="verified",
        production_training_allowed=True,
        blockers=[],
    )
    assert require_verified_production_target_binding(binding) is binding
    assert binding.total_parameters == "397B"
    assert binding.active_parameters == "35B"


def test_verified_binding_rejects_zero_revision_placeholder() -> None:
    binding = _binding(
        model_revision=_ZERO,
        verification_status="verified",
        production_training_allowed=True,
        blockers=[],
    )
    with pytest.raises(ValueError, match="zero placeholder"):
        require_verified_production_target_binding(binding)


def test_verified_binding_rejects_unwired_model_ref() -> None:
    binding = _binding(
        model_ref="UNWIRED-397B-35B-TARGET",
        verification_status="verified",
        production_training_allowed=True,
        blockers=[],
    )
    with pytest.raises(ValueError, match="UNWIRED placeholder"):
        require_verified_production_target_binding(binding)
