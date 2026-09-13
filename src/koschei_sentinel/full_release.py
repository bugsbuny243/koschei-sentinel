from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_authority import (
    ProductionAuthority,
    ProductionAuthorityBlocked,
    ProductionAuthorityProposal,
    canonical_json_digest,
    verify_production_authority,
)
from koschei_sentinel.promotion import public_key_fingerprint

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_APPROVER_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-full-release-v1\0"


class FullReleaseBlocked(ValueError):
    """Raised when canary evidence is insufficient for a full release."""


class CanaryEvidence(StrictModel):
    schema_version: Literal["sentinel.canary-evidence.v1"] = "sentinel.canary-evidence.v1"
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    production_authority_digest: str = Field(pattern=_DIGEST)
    observed_request_count: int = Field(ge=100)
    error_rate: float = Field(ge=0.0, le=1.0)
    critical_incident_count: int = Field(ge=0)
    rollback_drill_passed: bool
    emergency_disable_drill_passed: bool
    telemetry_complete: bool


class FullReleaseProposal(StrictModel):
    schema_version: Literal["sentinel.full-release-proposal.v1"] = (
        "sentinel.full-release-proposal.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    production_authority_digest: str = Field(pattern=_DIGEST)
    canary_evidence_digest: str = Field(pattern=_DIGEST)
    rollout_scope: Literal["general_release"] = "general_release"
    rollback_required: Literal[True] = True
    emergency_disable_required: Literal[True] = True
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)


class FullReleaseAuthority(StrictModel):
    schema_version: Literal["sentinel.full-release-authority.v1"] = (
        "sentinel.full-release-authority.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    proposal_digest: str = Field(pattern=_DIGEST)
    approver_id: str = Field(pattern=_APPROVER_ID)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    rollout_scope: Literal["general_release"] = "general_release"
    production_deployment_allowed: Literal[True] = True
    rollback_required: Literal[True] = True
    emergency_disable_required: Literal[True] = True
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    authority_digest: str = Field(pattern=_DIGEST)


def load_canary_evidence(path: str | Path) -> CanaryEvidence:
    try:
        return CanaryEvidence.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FullReleaseBlocked("invalid canary evidence") from exc


def _validate_canary(evidence: CanaryEvidence) -> None:
    if evidence.error_rate > 0.02:
        raise FullReleaseBlocked("canary error rate exceeds 2% release ceiling")
    if evidence.critical_incident_count != 0:
        raise FullReleaseBlocked("canary recorded a critical incident")
    if not evidence.rollback_drill_passed:
        raise FullReleaseBlocked("rollback drill did not pass")
    if not evidence.emergency_disable_drill_passed:
        raise FullReleaseBlocked("emergency-disable drill did not pass")
    if not evidence.telemetry_complete:
        raise FullReleaseBlocked("canary telemetry is incomplete")


def build_full_release_proposal(
    canary_proposal: ProductionAuthorityProposal,
    canary_authority: ProductionAuthority,
    canary_evidence: CanaryEvidence,
    owner_public_key: Ed25519PublicKey,
) -> FullReleaseProposal:
    _validate_canary(canary_evidence)
    try:
        verify_production_authority(canary_proposal, canary_authority, owner_public_key)
    except ProductionAuthorityBlocked as exc:
        raise FullReleaseBlocked("canary production authority verification failed") from exc
    if canary_evidence.candidate_id != canary_authority.candidate_id:
        raise FullReleaseBlocked("canary evidence candidate does not match production authority")
    if canary_evidence.production_authority_digest != canary_authority.authority_digest:
        raise FullReleaseBlocked("canary evidence does not bind the production authority")
    if canary_authority.deployment_scope != "canary_only":
        raise FullReleaseBlocked("full release requires a verified canary-only authority")
    payload = {
        "schema_version": "sentinel.full-release-proposal.v1",
        "candidate_id": canary_authority.candidate_id,
        "production_authority_digest": canary_authority.authority_digest,
        "canary_evidence_digest": canonical_json_digest(canary_evidence.model_dump(mode="json")),
        "rollout_scope": "general_release",
        "rollback_required": True,
        "emergency_disable_required": True,
        "owner_key_fingerprint": public_key_fingerprint(owner_public_key),
    }
    return FullReleaseProposal.model_validate(
        {**payload, "proposal_digest": canonical_json_digest(payload)}
    )


def approve_full_release(
    proposal: FullReleaseProposal,
    owner_private_key: Ed25519PrivateKey,
    *,
    approver_id: str,
) -> FullReleaseAuthority:
    fingerprint = public_key_fingerprint(owner_private_key.public_key())
    if fingerprint != proposal.owner_key_fingerprint:
        raise FullReleaseBlocked("private key does not match full-release proposal")
    signature = owner_private_key.sign(_signature_message(proposal.proposal_digest))
    payload = {
        "schema_version": "sentinel.full-release-authority.v1",
        "candidate_id": proposal.candidate_id,
        "proposal_digest": proposal.proposal_digest,
        "approver_id": approver_id,
        "owner_key_fingerprint": fingerprint,
        "rollout_scope": "general_release",
        "production_deployment_allowed": True,
        "rollback_required": True,
        "emergency_disable_required": True,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    return FullReleaseAuthority.model_validate(
        {**payload, "authority_digest": canonical_json_digest(payload)}
    )


def verify_full_release(
    proposal: FullReleaseProposal,
    authority: FullReleaseAuthority,
    owner_public_key: Ed25519PublicKey,
) -> FullReleaseAuthority:
    proposal_payload = proposal.model_dump(mode="json")
    claimed_proposal = proposal_payload.pop("proposal_digest")
    if canonical_json_digest(proposal_payload) != claimed_proposal:
        raise FullReleaseBlocked("full-release proposal digest mismatch")
    authority_payload = authority.model_dump(mode="json")
    claimed_authority = authority_payload.pop("authority_digest")
    if canonical_json_digest(authority_payload) != claimed_authority:
        raise FullReleaseBlocked("full-release authority digest mismatch")
    fingerprint = public_key_fingerprint(owner_public_key)
    if fingerprint != proposal.owner_key_fingerprint or fingerprint != authority.owner_key_fingerprint:
        raise FullReleaseBlocked("owner public key fingerprint mismatch")
    if authority.candidate_id != proposal.candidate_id:
        raise FullReleaseBlocked("full-release authority candidate mismatch")
    if authority.proposal_digest != proposal.proposal_digest:
        raise FullReleaseBlocked("full-release authority does not bind proposal")
    try:
        signature = base64.b64decode(authority.signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(proposal.proposal_digest))
    except (InvalidSignature, ValueError) as exc:
        raise FullReleaseBlocked("full-release owner signature verification failed") from exc
    return authority


def _signature_message(proposal_digest: str) -> bytes:
    return _SIGNATURE_CONTEXT + proposal_digest.encode("ascii")