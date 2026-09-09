import pytest

from koschei_sentinel.fabric_case_adapter_v1 import build_sentinel_fabric_projection_v1


def _projection(**overrides):
    values = {
        "interpretation_state": "AVAILABLE",
        "supporting_evidence_ids": ("evidence:1",),
        "uncertainty": ("fixture-only",),
        "native_ref": "fixture://sentinel/event/1",
        "native_digest_sha256": "b" * 64,
    }
    values.update(overrides)
    return build_sentinel_fabric_projection_v1(**values)


def test_projection_is_evidence_interpretation_only() -> None:
    projection = _projection()
    assert projection.sentinel["role"] == "evidence-interpretation-only"
    assert "authority" not in projection.sentinel
    assert projection.native_binding["owner"] == "koschei-sentinel"
    assert projection.native_binding["nativeSchema"] == "sentinel.web4-security-event.v1"


def test_available_interpretation_requires_evidence() -> None:
    with pytest.raises(ValueError, match="supporting evidence"):
        _projection(supporting_evidence_ids=())


def test_checkpoint_requires_model_identity() -> None:
    with pytest.raises(ValueError, match="model_ref"):
        _projection(checkpoint_digest_sha256="c" * 64)


def test_invalid_native_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _projection(native_digest_sha256="bad")


def test_duplicate_evidence_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        _projection(supporting_evidence_ids=("evidence:1", "evidence:1"))


def test_timeout_can_remain_evidence_empty() -> None:
    projection = _projection(
        interpretation_state="TIMEOUT",
        supporting_evidence_ids=(),
        uncertainty=("sentinel-timeout",),
    )
    assert projection.sentinel["interpretationState"] == "TIMEOUT"
    assert projection.sentinel["supportingEvidenceIds"] == []
