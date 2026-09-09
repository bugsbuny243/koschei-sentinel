from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.cyber_corpus_catalog import LicenseScope, LicenseStatus
from koschei_sentinel.pq_evidence_review import (
    PQEvidenceReviewProof,
    PQEvidenceReviewerTrustPolicy,
    build_pq_evidence_review_proof,
    build_pq_evidence_reviewer_trust_policy,
    write_pq_evidence_review_proof,
    write_pq_evidence_reviewer_trust_policy,
)
from koschei_sentinel.pq_network_intelligence import (
    build_pq_research_snapshot_receipt,
    materialize_pq_network_research_record,
    write_pq_research_snapshot_receipt,
)
from koschei_sentinel.pq_reviewed_evidence_admission import (
    PQReviewedEvidenceAdmissionPolicy,
    PQReviewedEvidenceAdmissionRequest,
    PQReviewedEvidenceCatalogEntry,
    build_pq_reviewed_evidence_catalog_entry,
    load_pq_reviewed_evidence_admission_policy,
    verify_pq_reviewed_evidence_catalog_entry,
    write_pq_reviewed_evidence_catalog_entry,
)

_ROOT = Path(__file__).resolve().parents[1]
_REVIEW_FIXTURE = _ROOT / "fixtures/pq/evidence-review"
_WATCH = _REVIEW_FIXTURE / "synthetic-watch.json"
_SNAPSHOT = _REVIEW_FIXTURE / "synthetic-source.txt"
_CLAIM = _REVIEW_FIXTURE / "synthetic-claim.json"
_REVIEW = _REVIEW_FIXTURE / "synthetic-review.json"
_REQUEST = _ROOT / "fixtures/pq/reviewed-catalog-admission/synthetic-request.json"
_POLICY = _ROOT / "configs/corpus/pq-reviewed-evidence-admission.v1.json"


def _prepare(
    tmp_path: Path,
    *,
    review_path: Path = _REVIEW,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    Ed25519PrivateKey,
    Ed25519PrivateKey,
    PQEvidenceReviewerTrustPolicy,
    PQEvidenceReviewProof,
]:
    snapshot_receipt = build_pq_research_snapshot_receipt(
        source_id="fixture.pq.evidence-review",
        snapshot_path=_SNAPSHOT,
        captured_at="2026-09-09T06:00:00+03:00",
        watch_registry_path=_WATCH,
    )
    snapshot_receipt_path = tmp_path / "snapshot-receipt.json"
    write_pq_research_snapshot_receipt(snapshot_receipt, snapshot_receipt_path)

    record_path = tmp_path / "record.json"
    materialization_path = tmp_path / "materialization-receipt.json"
    materialize_pq_network_research_record(
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        output_path=record_path,
        materialization_receipt_path=materialization_path,
    )

    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    trust = build_pq_evidence_reviewer_trust_policy(
        reviewer_public_key=reviewer.public_key(),
        owner_private_key=owner,
        policy_id="fixture-pq-review-policy",
    )
    trust_path = tmp_path / "reviewer-trust.json"
    write_pq_evidence_reviewer_trust_policy(trust, trust_path)

    proof = build_pq_evidence_review_proof(
        review_input_path=review_path,
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
        reviewer_private_key=reviewer,
        trust_policy=trust,
        owner_public_key=owner.public_key(),
    )
    proof_path = tmp_path / "review-proof.json"
    write_pq_evidence_review_proof(proof, proof_path)
    return (
        snapshot_receipt_path,
        record_path,
        materialization_path,
        trust_path,
        proof_path,
        owner,
        reviewer,
        trust,
        proof,
    )


def _build(
    tmp_path: Path,
    *,
    request_path: Path = _REQUEST,
    review_path: Path = _REVIEW,
) -> tuple[PQReviewedEvidenceCatalogEntry, tuple[object, ...]]:
    prepared = _prepare(tmp_path, review_path=review_path)
    (
        snapshot_receipt_path,
        record_path,
        materialization_path,
        trust_path,
        proof_path,
        owner,
        reviewer,
        _trust,
        _proof,
    ) = prepared
    entry = build_pq_reviewed_evidence_catalog_entry(
        admission_request_path=request_path,
        admission_policy_path=_POLICY,
        review_proof_path=proof_path,
        trust_policy_path=trust_path,
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
        reviewer_public_key=reviewer.public_key(),
        owner_public_key=owner.public_key(),
    )
    return entry, prepared


def test_verified_evidence_enters_reviewed_catalog_without_training_authority(
    tmp_path: Path,
) -> None:
    entry, prepared = _build(tmp_path)
    (
        snapshot_receipt_path,
        record_path,
        materialization_path,
        trust_path,
        proof_path,
        owner,
        reviewer,
        _trust,
        _proof,
    ) = prepared

    assert entry.catalog_tier == "REVIEWED_RESEARCH_CATALOG"
    assert entry.evidence_verified is True
    assert entry.reviewed_catalog_admitted is True
    assert entry.dataset_admission_allowed is False
    assert entry.split_assignment is None
    assert entry.evaluation_authorization is False
    assert entry.training_authorization is False
    assert entry.gold_eligible is False
    assert entry.production_activation_allowed is False
    assert entry.license_status is LicenseStatus.REVIEW_REQUIRED

    verified = verify_pq_reviewed_evidence_catalog_entry(
        entry,
        admission_request_path=_REQUEST,
        admission_policy_path=_POLICY,
        review_proof_path=proof_path,
        trust_policy_path=trust_path,
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
        reviewer_public_key=reviewer.public_key(),
        owner_public_key=owner.public_key(),
    )
    assert verified == entry


def test_rejected_review_cannot_enter_reviewed_catalog(tmp_path: Path) -> None:
    payload = json.loads(_REVIEW.read_text(encoding="utf-8"))
    payload.update(
        {
            "decision": "REJECTED",
            "source_match_verified": False,
            "claim_supported": False,
            "evidence_confidence": 0.1,
        }
    )
    review_path = tmp_path / "rejected-review.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    prepared = _prepare(tmp_path, review_path=review_path)
    (
        snapshot_receipt_path,
        record_path,
        materialization_path,
        trust_path,
        proof_path,
        owner,
        reviewer,
        _trust,
        _proof,
    ) = prepared

    with pytest.raises(ValueError, match="requires VERIFIED evidence review proof"):
        build_pq_reviewed_evidence_catalog_entry(
            admission_request_path=_REQUEST,
            admission_policy_path=_POLICY,
            review_proof_path=proof_path,
            trust_policy_path=trust_path,
            claim_path=_CLAIM,
            snapshot_receipt_path=snapshot_receipt_path,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_WATCH,
            record_path=record_path,
            materialization_receipt_path=materialization_path,
            reviewer_public_key=reviewer.public_key(),
            owner_public_key=owner.public_key(),
        )


def test_blocked_license_request_is_rejected() -> None:
    payload = {
        "schema_version": "sentinel.pq-reviewed-evidence-admission-request.v1",
        "admitted_at": "2026-09-09T06:08:00+03:00",
        "license_status": "BLOCKED",
        "license_scope": "UNKNOWN",
        "license_reference": None,
        "benchmark_overlap_risk": "UNKNOWN",
        "notes": None,
    }
    with pytest.raises(ValueError, match="BLOCKED PQ evidence"):
        PQReviewedEvidenceAdmissionRequest.model_validate(payload)


def test_allow_training_license_does_not_grant_training_authority(tmp_path: Path) -> None:
    payload = json.loads(_REQUEST.read_text(encoding="utf-8"))
    payload.update(
        {
            "license_status": "ALLOW_TRAINING",
            "license_scope": "UNIFORM",
            "license_reference": "https://example.invalid/synthetic-license",
        }
    )
    request_path = tmp_path / "allow-training-license-request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    entry, _prepared = _build(tmp_path, request_path=request_path)

    assert entry.license_status is LicenseStatus.ALLOW_TRAINING
    assert entry.license_scope is LicenseScope.UNIFORM
    assert entry.training_authorization is False
    assert entry.dataset_admission_allowed is False
    assert entry.split_assignment is None


def test_training_capable_license_requires_resolved_scope() -> None:
    payload = json.loads(_REQUEST.read_text(encoding="utf-8"))
    payload.update(
        {
            "license_status": "ALLOW_WITH_ATTRIBUTION",
            "license_scope": "UNKNOWN",
            "license_reference": "https://example.invalid/synthetic-license",
        }
    )
    with pytest.raises(ValueError, match="requires resolved license_scope"):
        PQReviewedEvidenceAdmissionRequest.model_validate(payload)


def test_admission_policy_cannot_accept_blocked_license() -> None:
    payload = load_pq_reviewed_evidence_admission_policy(_POLICY).model_dump(mode="json")
    payload["accepted_license_statuses"].append("BLOCKED")
    with pytest.raises(ValueError, match="both accepted and blocked"):
        PQReviewedEvidenceAdmissionPolicy.model_validate(payload)


def test_wrong_reviewer_key_cannot_admit_verified_evidence(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    (
        snapshot_receipt_path,
        record_path,
        materialization_path,
        trust_path,
        proof_path,
        owner,
        _reviewer,
        _trust,
        _proof,
    ) = prepared
    other = Ed25519PrivateKey.generate()

    with pytest.raises(ValueError, match="does not match owner-signed trust policy"):
        build_pq_reviewed_evidence_catalog_entry(
            admission_request_path=_REQUEST,
            admission_policy_path=_POLICY,
            review_proof_path=proof_path,
            trust_policy_path=trust_path,
            claim_path=_CLAIM,
            snapshot_receipt_path=snapshot_receipt_path,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_WATCH,
            record_path=record_path,
            materialization_receipt_path=materialization_path,
            reviewer_public_key=other.public_key(),
            owner_public_key=owner.public_key(),
        )


def test_catalog_entry_tamper_is_rejected_by_self_hash(tmp_path: Path) -> None:
    entry, _prepared = _build(tmp_path)
    payload = entry.model_dump(mode="json")
    payload["training_authorization"] = True
    with pytest.raises(ValueError):
        PQReviewedEvidenceCatalogEntry.model_validate(payload)


def test_admission_id_is_deterministic_for_same_verified_identity(tmp_path: Path) -> None:
    entry, prepared = _build(tmp_path)
    payload = json.loads(_REQUEST.read_text(encoding="utf-8"))
    payload["admitted_at"] = "2026-09-09T06:09:00+03:00"
    request_path = tmp_path / "later-request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    (
        snapshot_receipt_path,
        record_path,
        materialization_path,
        trust_path,
        proof_path,
        owner,
        reviewer,
        _trust,
        _proof,
    ) = prepared
    later = build_pq_reviewed_evidence_catalog_entry(
        admission_request_path=request_path,
        admission_policy_path=_POLICY,
        review_proof_path=proof_path,
        trust_policy_path=trust_path,
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
        reviewer_public_key=reviewer.public_key(),
        owner_public_key=owner.public_key(),
    )
    assert later.admission_id == entry.admission_id
    assert later.entry_sha256 != entry.entry_sha256


def test_catalog_entry_write_is_immutable(tmp_path: Path) -> None:
    entry, _prepared = _build(tmp_path)
    output = tmp_path / "catalog-entry.json"
    write_pq_reviewed_evidence_catalog_entry(entry, output)
    with pytest.raises(FileExistsError, match="already exists"):
        write_pq_reviewed_evidence_catalog_entry(entry, output)
