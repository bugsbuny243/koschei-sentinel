from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from koschei_sentinel.production_model_source import (
    ProductionModelSourceIntake,
    _canonical_digest,
    load_production_model_source_intake,
)


def _payload() -> dict[str, object]:
    payload = {
        "schema_version": "sentinel.production-model-source-intake.v1",
        "model_ref": "koschei/sentinel-397b-35b",
        "model_revision": "a" * 64,
        "source_kind": "internal-artifact",
        "source_ref": "pinned://sentinel/base",
        "source_revision": "b" * 64,
        "source_payload_sha256": "c" * 64,
        "license_ref": "internal://license/reviewed",
        "architecture_evidence_sha256": "1" * 64,
        "router_evidence_sha256": "2" * 64,
        "expert_topology_evidence_sha256": "3" * 64,
        "tokenizer_template_evidence_sha256": "4" * 64,
        "framework_compatibility_evidence_sha256": "5" * 64,
        "independently_reviewed": True,
        "review_ref": "review://production-model-source/001",
    }
    payload["intake_sha256"] = _canonical_digest(payload)
    return payload


def test_valid_intake_self_hash_loads(tmp_path) -> None:
    payload = _payload()
    path = tmp_path / "intake.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    intake = load_production_model_source_intake(path)
    assert intake.model_ref == "koschei/sentinel-397b-35b"


def test_tampered_intake_is_rejected(tmp_path) -> None:
    payload = _payload()
    payload["source_ref"] = "pinned://tampered"
    path = tmp_path / "intake.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        load_production_model_source_intake(path)


def test_zero_revision_is_rejected() -> None:
    payload = _payload()
    payload["model_revision"] = "0" * 64
    with pytest.raises(ValidationError):
        ProductionModelSourceIntake.model_validate(payload)
