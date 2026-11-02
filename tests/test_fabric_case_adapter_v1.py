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


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"supporting_evidence_ids": tuple(f"evidence:{i}" for i in range(257))}, "at most 256"),
        ({"supporting_evidence_ids": ("e" * 161,)}, "at most 160"),
        ({"uncertainty": tuple(f"u{i}" for i in range(129))}, "at most 128"),
        ({"uncertainty": ("u" * 2001,)}, "at most 2000"),
        ({"native_ref": "r" * 1025}, "native_ref"),
        ({"native_schema": "s" * 257}, "native_schema"),
        ({"adapter_version": "v" * 129}, "adapter_version"),
        ({"model_ref": "m" * 513}, "model_ref"),
    ],
)
def test_projection_rejects_values_the_fabric_envelope_rejects(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _projection(**overrides)


def test_projection_accepts_values_at_the_fabric_envelope_bounds() -> None:
    projection = _projection(
        supporting_evidence_ids=tuple(f"e{i:03d}".ljust(160, "x") for i in range(256)),
        uncertainty=tuple("u" * 2000 for _ in range(128)),
        native_ref="r" * 1024,
        native_schema="s" * 256,
        adapter_version="v" * 128,
        model_ref="m" * 512,
    )
    assert len(projection.sentinel["supportingEvidenceIds"]) == 256
