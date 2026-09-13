from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field

from koschei_sentinel.candidate_finalization import (
    CandidateFinalization,
    CandidateFinalizationBlocked,
    verify_candidate_finalization,
)
from koschei_sentinel.gold_holdout_evaluation_evidence import GoldHoldoutEvaluationEvidence
from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import public_key_fingerprint

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_APPROVER_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-production-canary-authority-v1\0"


class ProductionAuthorityBlocked(ValueError):
    """Raised when production authority cannot be created or verified safely."""


class ProductionAuthorityProposal(StrictModel):
    schema_version: Literal["sentinel.production-authority-proposal.v1"] = (
        "sentinel.production-authority-proposal.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    finalization_digest: str = Field(pattern=_DIGEST)
    holdout_evidence_digests: list[str] = Field(min_length=1)
    deployment_scope: Literal["canary_only"] = "canary_only"
    max_initial_traffic_percent: int = Field(ge=1, le=10)
    rollback_required: Literal[True] = True
    emergency_disable_required: Literal[True] = True
    automatic_expansion_allowed: Literal[False] = False
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)


class ProductionAuthority(StrictModel):
    schema_version: Literal["sentinel.production-authority.v1"] = (
        "sentinel.production-authority.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    proposal_digest: str = Field(pattern=_DIGEST)
    approver_id: str = Field(pattern=_APPROVER_ID)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    deployment_scope: Literal["canary_only"] = "canary_only"
    max_initial_traffic_percent: int = Field(ge=1, le=10)
    rollback_required: Literal[True] = True
    emergency_disable_required: Literal[True] = True
    automatic_expansion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[True] = True
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    authority_digest: str = Field(pattern=_DIGEST)


def canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_file_digest(path: str | Path) -> str:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return canonical_json_digest(payload)


def load_verified_production_holdout(path: str | Path) -> GoldHoldoutEvaluationEvidence:
    try:
        evidence = GoldHoldoutEvaluationEvidence.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ProductionAuthorityBlocked(
            f"invalid Gold HOLDOUT evaluation evidence: {path}"
        ) from exc

    if not evidence.passed:
        raise ProductionAuthorityBlocked(f"Gold HOLDOUT evidence did not pass: {path}")
    if not evidence.inference_verification_valid or not evidence.complete_case_accounting:
        raise ProductionAuthorityBlocked(
            f"Gold HOLDOUT evidence is not complete and verified: {path}"
        )
    if evidence.failure_count != 0:
        raise ProductionAuthorityBlocked(f"Gold HOLDOUT evidence has inference failures: {path}")

    required_production_bindings = {
        "review_signature_audit_sha256": evidence.review_signature_audit_sha256,
        "candidate_training_binding_verification_sha256": (
            evidence.candidate_training_binding_verification_sha256
        ),
        "inference_pack_signature_proof_sha256": evidence.inference_pack_signature_proof_sha256,
        "reviewer_trust_policy_sha256": evidence.reviewer_trust_policy_sha256,
        "owner_key_fingerprint": evidence.owner_key_fingerprint,
    }
    missing = sorted(
        name for name, value in required_production_bindings.items() if value is None
    )
    if missing:
        raise ProductionAuthorityBlocked(
            f"Gold HOLDOUT evidence is missing production trust bindings {missing}: {path}"
        )

    payload = evidence.model_dump(mode="json")
    claimed = payload.pop("evidence_sha256")
    if canonical_json_digest(payload) != claimed:
        raise ProductionAuthorityBlocked(f"Gold HOLDOUT evidence self-hash mismatch: {path}")
    return evidence


def build_production_authority_proposal(
    finalization: CandidateFinalization,
    holdout_evidence_paths: list[str | Path],
    owner_public_key: Ed25519PublicKey,
    *,
    max_initial_traffic_percent: int = 5,
) -> ProductionAuthorityProposal:
    if not holdout_evidence_paths:
        raise ProductionAuthorityBlocked("at least one holdout evidence artifact is required")
    try:
        verify_candidate_finalization(finalization)
    except CandidateFinalizationBlocked as exc:
        raise ProductionAuthorityBlocked("candidate finalization verification failed") from exc
    if finalization.state != "finalized_incubation":
        raise ProductionAuthorityBlocked("candidate is not finalized for incubation")

    owner_fingerprint = public_key_fingerprint(owner_public_key)
    holdout_digests: list[str] = []
    for path in holdout_evidence_paths:
        evidence = load_verified_production_holdout(path)
        if evidence.owner_key_fingerprint != owner_fingerprint:
            raise ProductionAuthorityBlocked(
                f"Gold HOLDOUT evidence owner trust root does not match proposal owner: {path}"
            )
        if evidence.adapter_digest != finalization.adapter_digest:
            raise ProductionAuthorityBlocked(
                f"Gold HOLDOUT evidence adapter does not match finalized candidate: {path}"
            )
        holdout_digests.append(evidence.evidence_sha256)

    payload = {
        "schema_version": "sentinel.production-authority-proposal.v1",
        "candidate_id": finalization.candidate_id,
        "finalization_digest": finalization.finalization_digest,
        "holdout_evidence_digests": sorted(holdout_digests),
        "deployment_scope": "canary_only",
        "max_initial_traffic_percent": max_initial_traffic_percent,
        "rollback_required": True,
        "emergency_disable_required": True,
        "automatic_expansion_allowed": False,
        "owner_key_fingerprint": owner_fingerprint,
    }
    return ProductionAuthorityProposal.model_validate(
        {**payload, "proposal_digest": canonical_json_digest(payload)}
    )


def approve_production_authority(
    proposal: ProductionAuthorityProposal,
    owner_private_key: Ed25519PrivateKey,
    *,
    approver_id: str,
) -> ProductionAuthority:
    public_key = owner_private_key.public_key()
    fingerprint = public_key_fingerprint(public_key)
    if fingerprint != proposal.owner_key_fingerprint:
        raise ProductionAuthorityBlocked("private key does not match proposal owner key")
    signature = owner_private_key.sign(_signature_message(proposal.proposal_digest))
    payload = {
        "schema_version": "sentinel.production-authority.v1",
        "candidate_id": proposal.candidate_id,
        "proposal_digest": proposal.proposal_digest,
        "approver_id": approver_id,
        "owner_key_fingerprint": fingerprint,
        "deployment_scope": proposal.deployment_scope,
        "max_initial_traffic_percent": proposal.max_initial_traffic_percent,
        "rollback_required": True,
        "emergency_disable_required": True,
        "automatic_expansion_allowed": False,
        "production_deployment_allowed": True,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    return ProductionAuthority.model_validate(
        {**payload, "authority_digest": canonical_json_digest(payload)}
    )


def verify_production_authority(
    proposal: ProductionAuthorityProposal,
    authority: ProductionAuthority,
    owner_public_key: Ed25519PublicKey,
) -> ProductionAuthority:
    if canonical_json_digest(
        {k: v for k, v in proposal.model_dump(mode="json").items() if k != "proposal_digest"}
    ) != proposal.proposal_digest:
        raise ProductionAuthorityBlocked("production authority proposal digest mismatch")
    if canonical_json_digest(
        {k: v for k, v in authority.model_dump(mode="json").items() if k != "authority_digest"}
    ) != authority.authority_digest:
        raise ProductionAuthorityBlocked("production authority digest mismatch")
    fingerprint = public_key_fingerprint(owner_public_key)
    if fingerprint != proposal.owner_key_fingerprint or fingerprint != authority.owner_key_fingerprint:
        raise ProductionAuthorityBlocked("owner public key fingerprint mismatch")
    if authority.candidate_id != proposal.candidate_id:
        raise ProductionAuthorityBlocked("authority candidate does not match proposal")
    if authority.proposal_digest != proposal.proposal_digest:
        raise ProductionAuthorityBlocked("authority does not bind supplied proposal")
    try:
        signature = base64.b64decode(authority.signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(proposal.proposal_digest))
    except (InvalidSignature, ValueError) as exc:
        raise ProductionAuthorityBlocked("production authority signature verification failed") from exc
    return authority


def _signature_message(proposal_digest: str) -> bytes:
    return _SIGNATURE_CONTEXT + proposal_digest.encode("ascii")