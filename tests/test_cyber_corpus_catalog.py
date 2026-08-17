from __future__ import annotations

import pytest
from pydantic import ValidationError

from koschei_sentinel.cyber_corpus_catalog import (
    CyberArtifact,
    CyberSource,
    audit_artifacts,
    audit_catalog,
    load_catalog,
)


def approved_source(**overrides):
    payload = {
        "source_id": "official.security.source",
        "source_class": "OFFICIAL_SECURITY_GUIDANCE",
        "domain_families": ["INCIDENT_RESPONSE", "EVIDENCE_GROUNDING"],
        "provenance_tier": "T1_AUTHORITATIVE",
        "license_status": "ALLOW_TRAINING",
        "license_scope": "UNIFORM",
        "license_reference": "reviewed-license-record-001",
        "acquisition_mode": "VERSIONED_RELEASE",
        "canonical_locator": "catalog://official.security.source",
        "pinned_revision": "release-2026-08",
        "training_authorization": True,
        "eval_exclusion": True,
        "benchmark_overlap_risk": "LOW",
        "review_status": "APPROVED",
    }
    payload.update(overrides)
    return CyberSource.model_validate(payload)


def artifact(**overrides):
    payload = {
        "artifact_id": "official.security.source:artifact-001",
        "source_id": "official.security.source",
        "locator": "catalog://official.security.source/artifact-001",
        "source_revision": "release-2026-08",
        "source_snapshot_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "license_status": "ALLOW_TRAINING",
        "license_reference": "reviewed-license-record-001",
        "inherited_source_license": True,
        "training_authorization": True,
        "eval_exclusion": True,
        "benchmark_overlap_risk": "LOW",
    }
    payload.update(overrides)
    return CyberArtifact.model_validate(payload)


def test_approved_pinned_source_is_collectable():
    audit = audit_catalog([approved_source()])
    assert audit.ready_for_collection is True
    assert audit.training_authorized_sources == 1
    assert audit.violations == []


def test_unknown_license_cannot_be_training_authorized():
    with pytest.raises(ValidationError):
        approved_source(license_status="REVIEW_REQUIRED")


def test_unknown_license_scope_cannot_be_training_authorized():
    with pytest.raises(ValidationError):
        approved_source(license_scope="UNKNOWN")


def test_context_only_source_cannot_be_training_authorized():
    with pytest.raises(ValidationError):
        approved_source(provenance_tier="T3_CONTEXT_ONLY")


def test_high_benchmark_overlap_cannot_be_training_authorized():
    with pytest.raises(ValidationError):
        approved_source(benchmark_overlap_risk="HIGH")


def test_trainable_source_requires_canonical_locator():
    with pytest.raises(ValidationError):
        approved_source(canonical_locator=None)


def test_trainable_source_must_be_excluded_from_eval_material():
    source = approved_source(eval_exclusion=False)
    audit = audit_catalog([source])
    assert audit.ready_for_collection is False
    assert audit.violations == [
        "training source official.security.source is not marked eval_exclusion"
    ]


def test_uniform_license_artifact_must_inherit_source_license():
    source = approved_source()
    result = audit_artifacts([artifact(inherited_source_license=False)], [source])
    assert result.ready_for_ingestion is False
    assert "must inherit uniform source license" in result.violations[0]


def test_per_artifact_source_requires_explicit_artifact_license():
    source = approved_source(license_scope="PER_ARTIFACT")
    good = artifact(
        inherited_source_license=False,
        license_status="ALLOW_WITH_ATTRIBUTION",
        license_reference="artifact-license-record-001",
    )
    result = audit_artifacts([good], [source])
    assert result.ready_for_ingestion is True
    assert result.training_authorized_artifacts == 1


def test_per_artifact_source_cannot_blindly_inherit_source_license():
    source = approved_source(license_scope="PER_ARTIFACT")
    result = audit_artifacts([artifact(inherited_source_license=True)], [source])
    assert result.ready_for_ingestion is False
    assert "cannot inherit per-artifact source license" in result.violations[0]


def test_artifact_revision_must_match_source_pin():
    result = audit_artifacts(
        [artifact(source_revision="different-revision")],
        [approved_source()],
    )
    assert result.ready_for_ingestion is False
    assert "revision does not match source pin" in result.violations[0]


def test_duplicate_artifact_content_is_rejected():
    first = artifact()
    second = artifact(
        artifact_id="official.security.source:artifact-002",
        locator="catalog://official.security.source/artifact-002",
    )
    result = audit_artifacts([first, second], [approved_source()])
    assert result.ready_for_ingestion is False
    assert result.duplicate_content_sha256 == ["b" * 64]


def test_first_approved_catalog_is_collectable():
    sources = load_catalog("configs/corpus/cyber-v3.sources.approved.jsonl")
    audit = audit_catalog(sources)
    assert audit.ready_for_collection is True
    assert audit.training_authorized_sources == 3
    assert audit.approved_sources == 3
    assert audit.license_scope_counts == {"PER_ARTIFACT": 1, "UNIFORM": 2}
