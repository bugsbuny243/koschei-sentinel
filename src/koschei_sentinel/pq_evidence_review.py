from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.pq_network_intelligence import (
    PQNetworkIntelligenceRecord,
    PQNetworkMaterializationReceipt,
    PQResearchSnapshotReceipt,
    verify_pq_network_materialization,
)
from koschei_sentinel.research_snapshot import (
    load_strict_json_object,
    parse_timezone_timestamp,
    sha256_canonical_json,
)
from koschei_sentinel.training import atomic_write

_DIGEST = r"^[a-f0-9]{64}$"
_POLICY_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_REVIEWER_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_TRUST_SIGNATURE_CONTEXT = b"koschei-sentinel-pq-reviewer-trust-v1\0"
_REVIEW_SIGNATURE_CONTEXT = b"koschei-sentinel-pq-evidence-review-v1\0"


class PQEvidenceReviewerTrustPolicy(StrictModel):
    schema_version: Literal["sentinel.pq-evidence-reviewer-trust-policy.v1"] = (
        "sentinel.pq-evidence-reviewer-trust-policy.v1"
    )
    policy_id: str = Field(pattern=_POLICY_ID)
    state: Literal["active"] = "active"
    authority: Literal["pq_evidence_review_signing_only"] = "pq_evidence_review_signing_only"
    review_scope: Literal["sentinel.pq-network-intelligence.v1"] = (
        "sentinel.pq-network-intelligence.v1"
    )
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    human_review_signature_required: Literal[True] = True
    source_match_review_required: Literal[True] = True
    claim_support_review_required: Literal[True] = True
    automatic_training_authorization_allowed: Literal[False] = False
    automatic_gold_admission_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    policy_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    owner_signature_base64: str = Field(min_length=80, max_length=128)
    owner_signature_verified: Literal[True] = True


class PQEvidenceReviewInput(StrictModel):
    schema_version: Literal["sentinel.pq-evidence-review-input.v1"] = (
        "sentinel.pq-evidence-review-input.v1"
    )
    reviewer_id: str = Field(pattern=_REVIEWER_ID)
    reviewed_at: str = Field(min_length=10, max_length=64)
    decision: Literal["VERIFIED", "REJECTED"]
    source_match_verified: bool
    claim_supported: bool
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: str | None = Field(default=None, max_length=4096)
    notes: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def review_contract_verifies(self) -> PQEvidenceReviewInput:
        parse_timezone_timestamp(self.reviewed_at, "PQ evidence reviewed_at")
        if self.decision == "VERIFIED" and not (
            self.source_match_verified and self.claim_supported
        ):
            raise ValueError(
                "PQ VERIFIED review requires source_match_verified and claim_supported"
            )
        return self


class PQEvidenceReviewProof(StrictModel):
    schema_version: Literal["sentinel.pq-evidence-review-proof.v1"] = (
        "sentinel.pq-evidence-review-proof.v1"
    )
    policy_id: str = Field(pattern=_POLICY_ID)
    trust_policy_digest: str = Field(pattern=_DIGEST)
    reviewer_id: str = Field(pattern=_REVIEWER_ID)
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    reviewed_at: str = Field(min_length=10, max_length=64)
    record_id: str = Field(min_length=8, max_length=256)
    source_id: str = Field(min_length=3, max_length=256)
    canonical_locator: str = Field(min_length=1, max_length=4096)
    claim_sha256: str = Field(pattern=_DIGEST)
    record_sha256: str = Field(pattern=_DIGEST)
    materialization_receipt_sha256: str = Field(pattern=_DIGEST)
    snapshot_receipt_sha256: str = Field(pattern=_DIGEST)
    snapshot_sha256: str = Field(pattern=_DIGEST)
    source_row_sha256: str = Field(pattern=_DIGEST)
    decision: Literal["VERIFIED", "REJECTED"]
    source_match_verified: bool
    claim_supported: bool
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: str | None = Field(default=None, max_length=4096)
    notes: str | None = Field(default=None, max_length=4096)
    evidence_verified: bool
    training_authorization: Literal[False] = False
    dataset_admission_allowed: Literal[False] = False
    gold_eligible: Literal[False] = False
    binding_sha256: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    proof_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def proof_contract_verifies(self) -> PQEvidenceReviewProof:
        parse_timezone_timestamp(self.reviewed_at, "PQ evidence reviewed_at")
        expected = (
            self.decision == "VERIFIED"
            and self.source_match_verified
            and self.claim_supported
        )
        if self.evidence_verified is not expected:
            raise ValueError("PQ evidence_verified does not match the signed review decision")
        return self


def _fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def load_pq_reviewer_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid PQ reviewer public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("PQ reviewer public key must be Ed25519")
    return key


def load_pq_reviewer_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, ValueError) as exc:
        raise ValueError("invalid PQ reviewer private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("PQ reviewer private key must be unencrypted Ed25519 PEM")
    return key


def load_pq_owner_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid PQ owner public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("PQ owner public key must be Ed25519")
    return key


def load_pq_owner_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, ValueError) as exc:
        raise ValueError("invalid PQ owner private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("PQ owner private key must be unencrypted Ed25519 PEM")
    return key


def _trust_payload(
    *,
    policy_id: str,
    reviewer_key_fingerprint: str,
    owner_key_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.pq-evidence-reviewer-trust-policy.v1",
        "policy_id": policy_id,
        "state": "active",
        "authority": "pq_evidence_review_signing_only",
        "review_scope": "sentinel.pq-network-intelligence.v1",
        "reviewer_key_fingerprint": reviewer_key_fingerprint,
        "owner_key_fingerprint": owner_key_fingerprint,
        "human_review_signature_required": True,
        "source_match_review_required": True,
        "claim_support_review_required": True,
        "automatic_training_authorization_allowed": False,
        "automatic_gold_admission_allowed": False,
        "production_deployment_allowed": False,
    }


def _trust_message(policy_digest: str) -> bytes:
    return _TRUST_SIGNATURE_CONTEXT + policy_digest.encode("ascii")


def build_pq_evidence_reviewer_trust_policy(
    *,
    reviewer_public_key: Ed25519PublicKey,
    owner_private_key: Ed25519PrivateKey,
    policy_id: str,
) -> PQEvidenceReviewerTrustPolicy:
    payload = _trust_payload(
        policy_id=policy_id,
        reviewer_key_fingerprint=_fingerprint(reviewer_public_key),
        owner_key_fingerprint=_fingerprint(owner_private_key.public_key()),
    )
    policy_digest = sha256_canonical_json(payload)
    signature = owner_private_key.sign(_trust_message(policy_digest))
    policy = PQEvidenceReviewerTrustPolicy.model_validate(
        {
            **payload,
            "policy_digest": policy_digest,
            "signature_algorithm": "ed25519",
            "owner_signature_base64": base64.b64encode(signature).decode("ascii"),
            "owner_signature_verified": True,
        }
    )
    return verify_pq_evidence_reviewer_trust_policy(
        policy,
        reviewer_public_key=reviewer_public_key,
        owner_public_key=owner_private_key.public_key(),
    )


def verify_pq_evidence_reviewer_trust_policy(
    policy: PQEvidenceReviewerTrustPolicy,
    *,
    reviewer_public_key: Ed25519PublicKey,
    owner_public_key: Ed25519PublicKey,
) -> PQEvidenceReviewerTrustPolicy:
    expected = _trust_payload(
        policy_id=policy.policy_id,
        reviewer_key_fingerprint=policy.reviewer_key_fingerprint,
        owner_key_fingerprint=policy.owner_key_fingerprint,
    )
    if sha256_canonical_json(expected) != policy.policy_digest:
        raise ValueError("PQ reviewer trust policy digest does not verify")
    if _fingerprint(reviewer_public_key) != policy.reviewer_key_fingerprint:
        raise ValueError("PQ reviewer public key does not match owner-signed trust policy")
    if _fingerprint(owner_public_key) != policy.owner_key_fingerprint:
        raise ValueError("PQ owner public key does not match reviewer trust policy")
    try:
        signature = base64.b64decode(policy.owner_signature_base64, validate=True)
        owner_public_key.verify(signature, _trust_message(policy.policy_digest))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("PQ reviewer trust policy owner signature verification failed") from exc
    return policy


def load_pq_evidence_reviewer_trust_policy(
    path: str | Path,
) -> PQEvidenceReviewerTrustPolicy:
    payload = load_strict_json_object(path, "PQ evidence reviewer trust policy")
    return PQEvidenceReviewerTrustPolicy.model_validate(payload)


def write_pq_evidence_reviewer_trust_policy(
    policy: PQEvidenceReviewerTrustPolicy,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"PQ reviewer trust policy already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _load_review_input(path: str | Path) -> PQEvidenceReviewInput:
    payload = load_strict_json_object(path, "PQ evidence review input")
    return PQEvidenceReviewInput.model_validate(payload)


def _load_materialization_receipt(path: str | Path) -> PQNetworkMaterializationReceipt:
    payload = load_strict_json_object(path, "PQ network materialization receipt")
    return PQNetworkMaterializationReceipt.model_validate(payload)


def _load_snapshot_receipt(path: str | Path) -> PQResearchSnapshotReceipt:
    payload = load_strict_json_object(path, "PQ research snapshot receipt")
    return PQResearchSnapshotReceipt.model_validate(payload)


def _load_record(path: str | Path) -> PQNetworkIntelligenceRecord:
    payload = load_strict_json_object(path, "PQ network intelligence record")
    return PQNetworkIntelligenceRecord.model_validate(payload)


def _review_binding_payload(
    *,
    review: PQEvidenceReviewInput,
    policy: PQEvidenceReviewerTrustPolicy,
    materialization: PQNetworkMaterializationReceipt,
    snapshot_receipt: PQResearchSnapshotReceipt,
    record: PQNetworkIntelligenceRecord,
) -> dict[str, object]:
    evidence_verified = (
        review.decision == "VERIFIED"
        and review.source_match_verified
        and review.claim_supported
    )
    return {
        "policy_id": policy.policy_id,
        "trust_policy_digest": policy.policy_digest,
        "reviewer_id": review.reviewer_id,
        "reviewer_key_fingerprint": policy.reviewer_key_fingerprint,
        "owner_key_fingerprint": policy.owner_key_fingerprint,
        "reviewed_at": review.reviewed_at,
        "record_id": record.record_id,
        "source_id": materialization.source_id,
        "canonical_locator": snapshot_receipt.canonical_locator,
        "claim_sha256": materialization.claim_sha256,
        "record_sha256": materialization.record_sha256,
        "materialization_receipt_sha256": materialization.receipt_sha256,
        "snapshot_receipt_sha256": snapshot_receipt.receipt_sha256,
        "snapshot_sha256": snapshot_receipt.snapshot_sha256,
        "source_row_sha256": snapshot_receipt.source_row_sha256,
        "decision": review.decision,
        "source_match_verified": review.source_match_verified,
        "claim_supported": review.claim_supported,
        "evidence_confidence": review.evidence_confidence,
        "uncertainty": review.uncertainty,
        "notes": review.notes,
        "evidence_verified": evidence_verified,
        "training_authorization": False,
        "dataset_admission_allowed": False,
        "gold_eligible": False,
    }


def _review_message(binding_sha256: str) -> bytes:
    return _REVIEW_SIGNATURE_CONTEXT + binding_sha256.encode("ascii")


def build_pq_evidence_review_proof(
    *,
    review_input_path: str | Path,
    claim_path: str | Path,
    snapshot_receipt_path: str | Path,
    snapshot_path: str | Path,
    watch_registry_path: str | Path,
    record_path: str | Path,
    materialization_receipt_path: str | Path,
    reviewer_private_key: Ed25519PrivateKey,
    trust_policy: PQEvidenceReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
) -> PQEvidenceReviewProof:
    verify_pq_evidence_reviewer_trust_policy(
        trust_policy,
        reviewer_public_key=reviewer_private_key.public_key(),
        owner_public_key=owner_public_key,
    )
    verify_pq_network_materialization(
        claim_path=claim_path,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=snapshot_path,
        watch_registry_path=watch_registry_path,
        record_path=record_path,
        materialization_receipt_path=materialization_receipt_path,
    )
    review = _load_review_input(review_input_path)
    materialization = _load_materialization_receipt(materialization_receipt_path)
    snapshot_receipt = _load_snapshot_receipt(snapshot_receipt_path)
    record = _load_record(record_path)
    binding = _review_binding_payload(
        review=review,
        policy=trust_policy,
        materialization=materialization,
        snapshot_receipt=snapshot_receipt,
        record=record,
    )
    binding_sha = sha256_canonical_json(binding)
    signature = reviewer_private_key.sign(_review_message(binding_sha))
    payload: dict[str, object] = {
        "schema_version": "sentinel.pq-evidence-review-proof.v1",
        **binding,
        "binding_sha256": binding_sha,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
    }
    payload["proof_sha256"] = sha256_canonical_json(payload)
    proof = PQEvidenceReviewProof.model_validate(payload)
    return verify_pq_evidence_review_proof(
        proof,
        claim_path=claim_path,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=snapshot_path,
        watch_registry_path=watch_registry_path,
        record_path=record_path,
        materialization_receipt_path=materialization_receipt_path,
        reviewer_public_key=reviewer_private_key.public_key(),
        trust_policy=trust_policy,
        owner_public_key=owner_public_key,
    )


def verify_pq_evidence_review_proof(
    proof: PQEvidenceReviewProof,
    *,
    claim_path: str | Path,
    snapshot_receipt_path: str | Path,
    snapshot_path: str | Path,
    watch_registry_path: str | Path,
    record_path: str | Path,
    materialization_receipt_path: str | Path,
    reviewer_public_key: Ed25519PublicKey,
    trust_policy: PQEvidenceReviewerTrustPolicy,
    owner_public_key: Ed25519PublicKey,
) -> PQEvidenceReviewProof:
    verify_pq_evidence_reviewer_trust_policy(
        trust_policy,
        reviewer_public_key=reviewer_public_key,
        owner_public_key=owner_public_key,
    )
    verify_pq_network_materialization(
        claim_path=claim_path,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=snapshot_path,
        watch_registry_path=watch_registry_path,
        record_path=record_path,
        materialization_receipt_path=materialization_receipt_path,
    )
    payload = proof.model_dump(mode="json")
    observed_proof_sha = str(payload.pop("proof_sha256"))
    if sha256_canonical_json(payload) != observed_proof_sha:
        raise ValueError("PQ evidence review proof self-hash does not verify")

    materialization = _load_materialization_receipt(materialization_receipt_path)
    snapshot_receipt = _load_snapshot_receipt(snapshot_receipt_path)
    record = _load_record(record_path)
    review = PQEvidenceReviewInput(
        reviewer_id=proof.reviewer_id,
        reviewed_at=proof.reviewed_at,
        decision=proof.decision,
        source_match_verified=proof.source_match_verified,
        claim_supported=proof.claim_supported,
        evidence_confidence=proof.evidence_confidence,
        uncertainty=proof.uncertainty,
        notes=proof.notes,
    )
    expected_binding = _review_binding_payload(
        review=review,
        policy=trust_policy,
        materialization=materialization,
        snapshot_receipt=snapshot_receipt,
        record=record,
    )
    if sha256_canonical_json(expected_binding) != proof.binding_sha256:
        raise ValueError("PQ evidence review binding digest does not verify")
    observed_binding = {
        key: getattr(proof, key)
        for key in expected_binding
    }
    if observed_binding != expected_binding:
        raise ValueError("PQ evidence review proof does not bind current evidence artifacts")
    try:
        signature = base64.b64decode(proof.signature_base64, validate=True)
        reviewer_public_key.verify(signature, _review_message(proof.binding_sha256))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("PQ evidence reviewer Ed25519 signature verification failed") from exc
    return proof


def load_pq_evidence_review_proof(path: str | Path) -> PQEvidenceReviewProof:
    payload = load_strict_json_object(path, "PQ evidence review proof")
    return PQEvidenceReviewProof.model_validate(payload)


def write_pq_evidence_review_proof(proof: PQEvidenceReviewProof, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"PQ evidence review proof already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(proof.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
