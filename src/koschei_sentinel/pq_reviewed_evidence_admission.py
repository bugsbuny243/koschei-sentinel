from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.cyber_corpus_catalog import LicenseScope, LicenseStatus
from koschei_sentinel.models import StrictModel
from koschei_sentinel.pq_evidence_review import (
    PQEvidenceReviewProof,
    PQEvidenceReviewerTrustPolicy,
    load_pq_evidence_review_proof,
    load_pq_evidence_reviewer_trust_policy,
    verify_pq_evidence_review_proof,
)
from koschei_sentinel.pq_network_intelligence import PQNetworkIntelligenceRecord
from koschei_sentinel.research_snapshot import (
    load_strict_json_object,
    parse_timezone_timestamp,
    sha256_canonical_json,
)
from koschei_sentinel.training import atomic_write

_DIGEST = r"^[a-f0-9]{64}$"
_POLICY_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_RESOLVED_LICENSES = {
    LicenseStatus.ALLOW_TRAINING,
    LicenseStatus.ALLOW_WITH_ATTRIBUTION,
    LicenseStatus.EVAL_ONLY,
}


class PQReviewedEvidenceAdmissionPolicy(StrictModel):
    schema_version: Literal["sentinel.pq-reviewed-evidence-admission-policy.v1"] = (
        "sentinel.pq-reviewed-evidence-admission-policy.v1"
    )
    policy_id: str = Field(pattern=_POLICY_ID)
    catalog_tier: Literal["REVIEWED_RESEARCH_CATALOG"] = "REVIEWED_RESEARCH_CATALOG"
    require_verified_review: Literal[True] = True
    require_owner_authorized_reviewer: Literal[True] = True
    accepted_license_statuses: list[LicenseStatus] = Field(min_length=1)
    blocked_license_statuses: list[LicenseStatus] = Field(min_length=1)
    resolved_license_requires_reference: Literal[True] = True
    resolved_training_license_requires_resolved_scope: Literal[True] = True
    split_assignment_allowed: Literal[False] = False
    evaluation_authorization_default: Literal[False] = False
    training_authorization_default: Literal[False] = False
    gold_eligibility_default: Literal[False] = False
    production_activation_allowed: Literal[False] = False

    @model_validator(mode="after")
    def admission_policy_is_fail_closed(self) -> PQReviewedEvidenceAdmissionPolicy:
        accepted = set(self.accepted_license_statuses)
        blocked = set(self.blocked_license_statuses)
        if len(accepted) != len(self.accepted_license_statuses):
            raise ValueError("PQ admission accepted_license_statuses must be unique")
        if len(blocked) != len(self.blocked_license_statuses):
            raise ValueError("PQ admission blocked_license_statuses must be unique")
        if accepted & blocked:
            raise ValueError("PQ admission license status cannot be both accepted and blocked")
        if LicenseStatus.BLOCKED not in blocked:
            raise ValueError("PQ admission policy must block BLOCKED license status")
        if LicenseStatus.BLOCKED in accepted:
            raise ValueError("PQ admission policy cannot accept BLOCKED license status")
        return self


class PQReviewedEvidenceAdmissionRequest(StrictModel):
    schema_version: Literal["sentinel.pq-reviewed-evidence-admission-request.v1"] = (
        "sentinel.pq-reviewed-evidence-admission-request.v1"
    )
    admitted_at: str = Field(min_length=10, max_length=64)
    license_status: LicenseStatus
    license_scope: LicenseScope = LicenseScope.UNKNOWN
    license_reference: str | None = Field(default=None, max_length=4096)
    benchmark_overlap_risk: Literal["NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "UNKNOWN"
    notes: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def request_is_well_formed(self) -> PQReviewedEvidenceAdmissionRequest:
        parse_timezone_timestamp(self.admitted_at, "PQ reviewed evidence admitted_at")
        if self.license_status is LicenseStatus.BLOCKED:
            raise ValueError("BLOCKED PQ evidence cannot enter the reviewed research catalog")
        if self.license_status in _RESOLVED_LICENSES and not self.license_reference:
            raise ValueError("resolved PQ license status requires license_reference")
        if (
            self.license_status
            in {LicenseStatus.ALLOW_TRAINING, LicenseStatus.ALLOW_WITH_ATTRIBUTION}
            and self.license_scope is LicenseScope.UNKNOWN
        ):
            raise ValueError("training-capable PQ license status requires resolved license_scope")
        return self


class PQReviewedEvidenceCatalogEntry(StrictModel):
    schema_version: Literal["sentinel.pq-reviewed-evidence-catalog-entry.v1"] = (
        "sentinel.pq-reviewed-evidence-catalog-entry.v1"
    )
    admission_id: str = Field(min_length=16, max_length=256)
    policy_id: str = Field(pattern=_POLICY_ID)
    policy_sha256: str = Field(pattern=_DIGEST)
    catalog_tier: Literal["REVIEWED_RESEARCH_CATALOG"] = "REVIEWED_RESEARCH_CATALOG"
    source_id: str = Field(min_length=3, max_length=256)
    record_id: str = Field(min_length=8, max_length=256)
    network: str = Field(min_length=2, max_length=256)
    source_class: str = Field(min_length=3, max_length=128)
    canonical_locator: str = Field(min_length=1, max_length=4096)
    claim_sha256: str = Field(pattern=_DIGEST)
    record_sha256: str = Field(pattern=_DIGEST)
    snapshot_sha256: str = Field(pattern=_DIGEST)
    review_proof_sha256: str = Field(pattern=_DIGEST)
    trust_policy_digest: str = Field(pattern=_DIGEST)
    reviewer_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    admitted_at: str = Field(min_length=10, max_length=64)
    license_status: LicenseStatus
    license_scope: LicenseScope
    license_reference: str | None = Field(default=None, max_length=4096)
    benchmark_overlap_risk: Literal["NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    evidence_verified: Literal[True] = True
    reviewed_catalog_admitted: Literal[True] = True
    dataset_admission_allowed: Literal[False] = False
    split_assignment: None = None
    evaluation_authorization: Literal[False] = False
    training_authorization: Literal[False] = False
    gold_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    notes: str | None = Field(default=None, max_length=4096)
    entry_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def entry_is_self_bound(self) -> PQReviewedEvidenceCatalogEntry:
        parse_timezone_timestamp(self.admitted_at, "PQ reviewed evidence admitted_at")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("entry_sha256"))
        if sha256_canonical_json(unsigned) != observed:
            raise ValueError("PQ reviewed evidence catalog entry self-hash does not verify")
        return self


def load_pq_reviewed_evidence_admission_policy(
    path: str | Path,
) -> PQReviewedEvidenceAdmissionPolicy:
    payload = load_strict_json_object(path, "PQ reviewed evidence admission policy")
    return PQReviewedEvidenceAdmissionPolicy.model_validate(payload)


def load_pq_reviewed_evidence_admission_request(
    path: str | Path,
) -> PQReviewedEvidenceAdmissionRequest:
    payload = load_strict_json_object(path, "PQ reviewed evidence admission request")
    return PQReviewedEvidenceAdmissionRequest.model_validate(payload)


def load_pq_reviewed_evidence_catalog_entry(
    path: str | Path,
) -> PQReviewedEvidenceCatalogEntry:
    payload = load_strict_json_object(path, "PQ reviewed evidence catalog entry")
    return PQReviewedEvidenceCatalogEntry.model_validate(payload)


def _load_record(path: str | Path) -> PQNetworkIntelligenceRecord:
    payload = load_strict_json_object(path, "PQ network intelligence record")
    return PQNetworkIntelligenceRecord.model_validate(payload)


def _validate_license(
    request: PQReviewedEvidenceAdmissionRequest,
    policy: PQReviewedEvidenceAdmissionPolicy,
) -> None:
    if request.license_status in set(policy.blocked_license_statuses):
        raise ValueError("PQ evidence license status is blocked by admission policy")
    if request.license_status not in set(policy.accepted_license_statuses):
        raise ValueError("PQ evidence license status is not accepted by admission policy")
    if (
        policy.resolved_license_requires_reference
        and request.license_status in _RESOLVED_LICENSES
        and not request.license_reference
    ):
        raise ValueError("PQ admission policy requires license_reference for resolved status")
    if (
        policy.resolved_training_license_requires_resolved_scope
        and request.license_status
        in {LicenseStatus.ALLOW_TRAINING, LicenseStatus.ALLOW_WITH_ATTRIBUTION}
        and request.license_scope is LicenseScope.UNKNOWN
    ):
        raise ValueError("PQ admission policy requires resolved license_scope")


def _catalog_payload(
    *,
    request: PQReviewedEvidenceAdmissionRequest,
    policy: PQReviewedEvidenceAdmissionPolicy,
    proof: PQEvidenceReviewProof,
    record: PQNetworkIntelligenceRecord,
) -> dict[str, object]:
    if proof.decision != "VERIFIED" or not proof.evidence_verified:
        raise ValueError("PQ reviewed catalog admission requires VERIFIED evidence review proof")
    if proof.training_authorization or proof.dataset_admission_allowed or proof.gold_eligible:
        raise ValueError("PQ evidence review proof exceeded its non-authorizing trust boundary")
    if record.record_id != proof.record_id:
        raise ValueError("PQ reviewed catalog record_id differs from review proof")
    if not record.evidence:
        raise ValueError("PQ reviewed catalog record has no evidence")
    if len(record.evidence) != 1:
        raise ValueError("PQ reviewed catalog v1 requires exactly one bound evidence row")
    evidence = record.evidence[0]
    if evidence.canonical_locator != proof.canonical_locator:
        raise ValueError("PQ reviewed catalog canonical locator differs from review proof")
    if evidence.snapshot_sha256 != proof.snapshot_sha256:
        raise ValueError("PQ reviewed catalog snapshot SHA differs from review proof")

    policy_sha = sha256_canonical_json(policy.model_dump(mode="json"))
    identity = sha256_canonical_json(
        {
            "policy_sha256": policy_sha,
            "record_sha256": proof.record_sha256,
            "review_proof_sha256": proof.proof_sha256,
        }
    )
    return {
        "schema_version": "sentinel.pq-reviewed-evidence-catalog-entry.v1",
        "admission_id": f"pq-reviewed:{record.record_id}:{identity[:20]}",
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha,
        "catalog_tier": policy.catalog_tier,
        "source_id": proof.source_id,
        "record_id": record.record_id,
        "network": record.network,
        "source_class": evidence.source_class,
        "canonical_locator": proof.canonical_locator,
        "claim_sha256": proof.claim_sha256,
        "record_sha256": proof.record_sha256,
        "snapshot_sha256": proof.snapshot_sha256,
        "review_proof_sha256": proof.proof_sha256,
        "trust_policy_digest": proof.trust_policy_digest,
        "reviewer_key_fingerprint": proof.reviewer_key_fingerprint,
        "owner_key_fingerprint": proof.owner_key_fingerprint,
        "admitted_at": request.admitted_at,
        "license_status": request.license_status.value,
        "license_scope": request.license_scope.value,
        "license_reference": request.license_reference,
        "benchmark_overlap_risk": request.benchmark_overlap_risk,
        "evidence_verified": True,
        "reviewed_catalog_admitted": True,
        "dataset_admission_allowed": False,
        "split_assignment": None,
        "evaluation_authorization": False,
        "training_authorization": False,
        "gold_eligible": False,
        "production_activation_allowed": False,
        "notes": request.notes,
    }


def build_pq_reviewed_evidence_catalog_entry(
    *,
    admission_request_path: str | Path,
    admission_policy_path: str | Path,
    review_proof_path: str | Path,
    trust_policy_path: str | Path,
    claim_path: str | Path,
    snapshot_receipt_path: str | Path,
    snapshot_path: str | Path,
    watch_registry_path: str | Path,
    record_path: str | Path,
    materialization_receipt_path: str | Path,
    reviewer_public_key: Ed25519PublicKey,
    owner_public_key: Ed25519PublicKey,
) -> PQReviewedEvidenceCatalogEntry:
    policy = load_pq_reviewed_evidence_admission_policy(admission_policy_path)
    request = load_pq_reviewed_evidence_admission_request(admission_request_path)
    _validate_license(request, policy)
    trust_policy: PQEvidenceReviewerTrustPolicy = load_pq_evidence_reviewer_trust_policy(
        trust_policy_path
    )
    proof = load_pq_evidence_review_proof(review_proof_path)
    verify_pq_evidence_review_proof(
        proof,
        claim_path=claim_path,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=snapshot_path,
        watch_registry_path=watch_registry_path,
        record_path=record_path,
        materialization_receipt_path=materialization_receipt_path,
        reviewer_public_key=reviewer_public_key,
        trust_policy=trust_policy,
        owner_public_key=owner_public_key,
    )
    record = _load_record(record_path)
    unsigned = _catalog_payload(request=request, policy=policy, proof=proof, record=record)
    return PQReviewedEvidenceCatalogEntry(
        **unsigned,
        entry_sha256=sha256_canonical_json(unsigned),
    )


def verify_pq_reviewed_evidence_catalog_entry(
    entry: PQReviewedEvidenceCatalogEntry,
    *,
    admission_request_path: str | Path,
    admission_policy_path: str | Path,
    review_proof_path: str | Path,
    trust_policy_path: str | Path,
    claim_path: str | Path,
    snapshot_receipt_path: str | Path,
    snapshot_path: str | Path,
    watch_registry_path: str | Path,
    record_path: str | Path,
    materialization_receipt_path: str | Path,
    reviewer_public_key: Ed25519PublicKey,
    owner_public_key: Ed25519PublicKey,
) -> PQReviewedEvidenceCatalogEntry:
    expected = build_pq_reviewed_evidence_catalog_entry(
        admission_request_path=admission_request_path,
        admission_policy_path=admission_policy_path,
        review_proof_path=review_proof_path,
        trust_policy_path=trust_policy_path,
        claim_path=claim_path,
        snapshot_receipt_path=snapshot_receipt_path,
        snapshot_path=snapshot_path,
        watch_registry_path=watch_registry_path,
        record_path=record_path,
        materialization_receipt_path=materialization_receipt_path,
        reviewer_public_key=reviewer_public_key,
        owner_public_key=owner_public_key,
    )
    if expected != entry:
        raise ValueError("PQ reviewed evidence catalog entry differs from current verified inputs")
    return entry


def write_pq_reviewed_evidence_catalog_entry(
    entry: PQReviewedEvidenceCatalogEntry,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"PQ reviewed evidence catalog entry already exists: {destination}")
    atomic_write(
        destination,
        json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
