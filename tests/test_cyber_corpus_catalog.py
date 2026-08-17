from __future__ import annotations

import pytest
from pydantic import ValidationError

from koschei_sentinel.cyber_corpus_catalog import CyberSource, audit_catalog, load_catalog


def approved_source(**overrides):
    payload = {
        "source_id": "official.security.source",
        "source_class": "OFFICIAL_SECURITY_GUIDANCE",
        "domain_families": ["INCIDENT_RESPONSE", "EVIDENCE_GROUNDING"],
        "provenance_tier": "T1_AUTHORITATIVE",
        "license_status": "ALLOW_TRAINING",
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


def test_approved_pinned_source_is_collectable():
    audit = audit_catalog([approved_source()])
    assert audit.ready_for_collection is True
    assert audit.training_authorized_sources == 1
    assert audit.violations == []


def test_unknown_license_cannot_be_training_authorized():
    with pytest.raises(ValidationError):
        approved_source(license_status="REVIEW_REQUIRED")


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


def test_first_approved_catalog_is_collectable():
    sources = load_catalog("configs/corpus/cyber-v3.sources.approved.jsonl")
    audit = audit_catalog(sources)
    assert audit.ready_for_collection is True
    assert audit.training_authorized_sources == 2
    assert audit.approved_sources == 2
