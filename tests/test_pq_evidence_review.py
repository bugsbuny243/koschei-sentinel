from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.pq_evidence_review import (
    PQEvidenceReviewerTrustPolicy,
    PQEvidenceReviewInput,
    PQEvidenceReviewProof,
    build_pq_evidence_review_proof,
    build_pq_evidence_reviewer_trust_policy,
    verify_pq_evidence_review_proof,
    verify_pq_evidence_reviewer_trust_policy,
)
from koschei_sentinel.pq_network_intelligence import (
    build_pq_research_snapshot_receipt,
    materialize_pq_network_research_record,
    write_pq_research_snapshot_receipt,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/pq/evidence-review"
_WATCH = _FIXTURE / "synthetic-watch.json"
_SNAPSHOT = _FIXTURE / "synthetic-source.txt"
_CLAIM = _FIXTURE / "synthetic-claim.json"
_REVIEW = _FIXTURE / "synthetic-review.json"
_CAPTURED_AT = "2026-09-09T05:56:30+03:00"


def _materialize(tmp_path: Path) -> tuple[Path, Path, Path]:
    snapshot_receipt = build_pq_research_snapshot_receipt(
        source_id="fixture.pq.evidence-review",
        snapshot_path=_SNAPSHOT,
        captured_at=_CAPTURED_AT,
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
    return snapshot_receipt_path, record_path, materialization_path


def _trust() -> tuple[Ed25519PrivateKey, Ed25519PrivateKey, PQEvidenceReviewerTrustPolicy]:
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    policy = build_pq_evidence_reviewer_trust_policy(
        reviewer_public_key=reviewer.public_key(),
        owner_private_key=owner,
        policy_id="fixture-pq-review-policy",
    )
    return owner, reviewer, policy


def _build_proof(
    tmp_path: Path,
    *,
    review_path: Path = _REVIEW,
) -> tuple[
    PQEvidenceReviewProof,
    Ed25519PrivateKey,
    Ed25519PrivateKey,
    PQEvidenceReviewerTrustPolicy,
    Path,
    Path,
    Path,
]:
    snapshot_receipt_path, record_path, materialization_path = _materialize(tmp_path)
    owner, reviewer, policy = _trust()
    proof = build_pq_evidence_review_proof(
        review_input_path=review_path,
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
        reviewer_private_key=reviewer,
        trust_policy=policy,
        owner_public_key=owner.public_key(),
    )
    return (
        proof,
        owner,
        reviewer,
        policy,
        snapshot_receipt_path,
        record_path,
        materialization_path,
    )


def test_verified_review_is_signed_but_never_training_authorizing(tmp_path: Path) -> None:
    (
        proof,
        owner,
        reviewer,
        policy,
        snapshot_receipt_path,
        record_path,
        materialization_path,
    ) = _build_proof(tmp_path)

    assert proof.decision == "VERIFIED"
    assert proof.source_match_verified is True
    assert proof.claim_supported is True
    assert proof.evidence_verified is True
    assert proof.training_authorization is False
    assert proof.dataset_admission_allowed is False
    assert proof.gold_eligible is False

    verified = verify_pq_evidence_review_proof(
        proof,
        claim_path=_CLAIM,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=_SNAPSHOT,
        watch_registry_path=_WATCH,
        record_path=record_path,
        materialization_receipt_path=materialization_path,
        reviewer_public_key=reviewer.public_key(),
        trust_policy=policy,
        owner_public_key=owner.public_key(),
    )
    assert verified == proof


def test_rejected_review_cannot_mark_evidence_verified(tmp_path: Path) -> None:
    review_payload = json.loads(_REVIEW.read_text(encoding="utf-8"))
    review_payload.update(
        {
            "decision": "REJECTED",
            "source_match_verified": False,
            "claim_supported": False,
            "evidence_confidence": 0.1,
        }
    )
    review_path = tmp_path / "rejected-review.json"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")

    proof, *_ = _build_proof(tmp_path, review_path=review_path)
    assert proof.decision == "REJECTED"
    assert proof.evidence_verified is False
    assert proof.training_authorization is False


def test_verified_review_requires_source_and_claim_confirmation() -> None:
    payload = json.loads(_REVIEW.read_text(encoding="utf-8"))
    payload["source_match_verified"] = False
    with pytest.raises(ValueError, match="requires source_match_verified and claim_supported"):
        PQEvidenceReviewInput.model_validate(payload)


def test_owner_signed_trust_policy_rejects_other_reviewer() -> None:
    owner, _reviewer, policy = _trust()
    other = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="does not match owner-signed trust policy"):
        verify_pq_evidence_reviewer_trust_policy(
            policy,
            reviewer_public_key=other.public_key(),
            owner_public_key=owner.public_key(),
        )


def test_trust_policy_owner_signature_tamper_is_rejected() -> None:
    owner, reviewer, policy = _trust()
    payload = policy.model_dump(mode="json")
    payload["owner_signature_base64"] = "A" * len(payload["owner_signature_base64"])
    tampered = PQEvidenceReviewerTrustPolicy.model_validate(payload)
    with pytest.raises(ValueError, match="owner signature verification failed"):
        verify_pq_evidence_reviewer_trust_policy(
            tampered,
            reviewer_public_key=reviewer.public_key(),
            owner_public_key=owner.public_key(),
        )


def test_review_proof_tamper_is_rejected(tmp_path: Path) -> None:
    (
        proof,
        owner,
        reviewer,
        policy,
        snapshot_receipt_path,
        record_path,
        materialization_path,
    ) = _build_proof(tmp_path)
    payload = proof.model_dump(mode="json")
    payload["notes"] = "tampered after signature"
    tampered = PQEvidenceReviewProof.model_validate(payload)

    with pytest.raises(ValueError, match="self-hash does not verify"):
        verify_pq_evidence_review_proof(
            tampered,
            claim_path=_CLAIM,
            snapshot_receipt_path=snapshot_receipt_path,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_WATCH,
            record_path=record_path,
            materialization_receipt_path=materialization_path,
            reviewer_public_key=reviewer.public_key(),
            trust_policy=policy,
            owner_public_key=owner.public_key(),
        )


def test_underlying_record_tamper_invalidates_review_proof(tmp_path: Path) -> None:
    (
        proof,
        owner,
        reviewer,
        policy,
        snapshot_receipt_path,
        record_path,
        materialization_path,
    ) = _build_proof(tmp_path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["confidence"] = 0.01
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="record differs from current claim/evidence binding"):
        verify_pq_evidence_review_proof(
            proof,
            claim_path=_CLAIM,
            snapshot_receipt_path=snapshot_receipt_path,
            snapshot_path=_SNAPSHOT,
            watch_registry_path=_WATCH,
            record_path=record_path,
            materialization_receipt_path=materialization_path,
            reviewer_public_key=reviewer.public_key(),
            trust_policy=policy,
            owner_public_key=owner.public_key(),
        )


def test_key_serialization_contract_uses_ed25519() -> None:
    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert b"PRIVATE KEY" in private_pem
    assert b"PUBLIC KEY" in public_pem
