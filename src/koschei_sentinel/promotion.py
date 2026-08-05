from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field

from koschei_sentinel.candidate_finalization import CandidateFinalization
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_APPROVER_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-shadow-promotion-v1\0"


class PromotionBlocked(ValueError):
    """Raised when a signed shadow-promotion artifact fails closed."""


class PromotionProposal(StrictModel):
    schema_version: Literal["sentinel.promotion-proposal.v1"] = (
        "sentinel.promotion-proposal.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    state: Literal["awaiting_owner_signature"] = "awaiting_owner_signature"
    requested_stage: Literal["shadow_research_candidate"] = (
        "shadow_research_candidate"
    )
    authority: Literal["explanation_only"] = "explanation_only"
    finalization_digest: str = Field(pattern=_DIGEST)
    adapter_digest: str = Field(pattern=_DIGEST)
    benchmark_suite_digest: str = Field(pattern=_DIGEST)
    benchmark_report_digest: str = Field(pattern=_DIGEST)
    registry_digest: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_signature_required: Literal[True] = True
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    proposal_digest: str = Field(pattern=_DIGEST)


class PromotionApproval(StrictModel):
    schema_version: Literal["sentinel.promotion-approval.v1"] = (
        "sentinel.promotion-approval.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    state: Literal["owner_approved_shadow_research"] = (
        "owner_approved_shadow_research"
    )
    approved_stage: Literal["shadow_research_candidate"] = (
        "shadow_research_candidate"
    )
    authority: Literal["explanation_only"] = "explanation_only"
    approver_id: str = Field(pattern=_APPROVER_ID)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    approval_digest: str = Field(pattern=_DIGEST)


def load_candidate_finalization(path: str | Path) -> CandidateFinalization:
    try:
        finalization = CandidateFinalization.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid candidate finalization") from exc
    _require_model_digest(finalization, "finalization_digest", "candidate finalization")
    return finalization


def load_promotion_proposal(path: str | Path) -> PromotionProposal:
    try:
        proposal = PromotionProposal.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid promotion proposal") from exc
    _require_model_digest(proposal, "proposal_digest", "promotion proposal")
    return proposal


def load_promotion_approval(path: str | Path) -> PromotionApproval:
    try:
        approval = PromotionApproval.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid promotion approval") from exc
    _require_model_digest(approval, "approval_digest", "promotion approval")
    return approval


def load_owner_public_key(path: str | Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("owner public key must be Ed25519")
    return key


def load_owner_private_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("owner private key must be unencrypted Ed25519 PEM")
    return key


def build_promotion_proposal(
    finalization: CandidateFinalization,
    owner_public_key: Ed25519PublicKey,
) -> PromotionProposal:
    _require_model_digest(finalization, "finalization_digest", "candidate finalization")
    if finalization.state != "finalized_incubation":
        raise PromotionBlocked("candidate is not finalized for incubation")
    if finalization.authority != "explanation_only":
        raise PromotionBlocked("candidate authority exceeds explanation-only scope")

    payload = {
        "schema_version": "sentinel.promotion-proposal.v1",
        "candidate_id": finalization.candidate_id,
        "state": "awaiting_owner_signature",
        "requested_stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "finalization_digest": finalization.finalization_digest,
        "adapter_digest": finalization.adapter_digest,
        "benchmark_suite_digest": finalization.benchmark_suite_digest,
        "benchmark_report_digest": finalization.benchmark_report_digest,
        "registry_digest": finalization.updated_registry_digest,
        "owner_key_fingerprint": public_key_fingerprint(owner_public_key),
        "owner_signature_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return PromotionProposal.model_validate(
        {**payload, "proposal_digest": _digest(payload)}
    )


def approve_promotion_proposal(
    proposal: PromotionProposal,
    owner_private_key: Ed25519PrivateKey,
    *,
    approver_id: str,
) -> PromotionApproval:
    _require_model_digest(proposal, "proposal_digest", "promotion proposal")
    public_key = owner_private_key.public_key()
    fingerprint = public_key_fingerprint(public_key)
    if fingerprint != proposal.owner_key_fingerprint:
        raise PromotionBlocked("private key does not match proposal owner key")

    signature = owner_private_key.sign(_signature_message(proposal.proposal_digest))
    public_key.verify(signature, _signature_message(proposal.proposal_digest))
    payload = {
        "schema_version": "sentinel.promotion-approval.v1",
        "candidate_id": proposal.candidate_id,
        "state": "owner_approved_shadow_research",
        "approved_stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "approver_id": approver_id,
        "owner_key_fingerprint": fingerprint,
        "proposal_digest": proposal.proposal_digest,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return PromotionApproval.model_validate(
        {**payload, "approval_digest": _digest(payload)}
    )


def verify_promotion_approval(
    proposal: PromotionProposal,
    approval: PromotionApproval,
    owner_public_key: Ed25519PublicKey,
) -> PromotionApproval:
    _require_model_digest(proposal, "proposal_digest", "promotion proposal")
    _require_model_digest(approval, "approval_digest", "promotion approval")
    fingerprint = public_key_fingerprint(owner_public_key)
    if proposal.owner_key_fingerprint != fingerprint:
        raise PromotionBlocked("public key does not match promotion proposal")
    if approval.owner_key_fingerprint != fingerprint:
        raise PromotionBlocked("public key does not match promotion approval")
    if approval.candidate_id != proposal.candidate_id:
        raise PromotionBlocked("approval candidate does not match proposal")
    if approval.proposal_digest != proposal.proposal_digest:
        raise PromotionBlocked("approval does not bind the supplied proposal")

    try:
        signature = base64.b64decode(approval.signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(proposal.proposal_digest))
    except (InvalidSignature, ValueError) as exc:
        raise PromotionBlocked("owner signature verification failed") from exc
    return approval


def public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def write_artifact(model: StrictModel, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"promotion artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_model_digest(model: StrictModel, field: str, label: str) -> None:
    payload = model.model_dump(mode="json")
    claimed = payload.pop(field)
    if claimed != _digest(payload):
        raise PromotionBlocked(f"{label} digest does not match its contents")


def _signature_message(proposal_digest: str) -> bytes:
    return _SIGNATURE_CONTEXT + proposal_digest.encode("ascii")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
