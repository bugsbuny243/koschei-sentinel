from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field

from koschei_sentinel.candidate_finalization import CandidateFinalization
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
    holdout_evidence_digests: list[str]
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


def build_production_authority_proposal(
    finalization: CandidateFinalization,
    holdout_evidence_paths: list[str | Path],
    owner_public_key: Ed25519PublicKey,
    *,
    max_initial_traffic_percent: int = 5,
) -> ProductionAuthorityProposal:
    if not holdout_evidence_paths:
        raise ProductionAuthorityBlocked("at least one holdout evidence artifact is required")
    if finalization.state != "finalized_incubation":
        raise ProductionAuthorityBlocked("candidate is not finalized for incubation")

    holdout_digests: list[str] = []
    for path in holdout_evidence_paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductionAuthorityBlocked(f"invalid holdout evidence: {path}") from exc
        schema = payload.get("schema_version") if isinstance(payload, dict) else None
        if not isinstance(schema, str) or "holdout" not in schema.lower():
            raise ProductionAuthorityBlocked(f"unrecognized holdout evidence schema: {path}")
        holdout_digests.append(canonical_json_digest(payload))

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
        "owner_key_fingerprint": public_key_fingerprint(owner_public_key),
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
