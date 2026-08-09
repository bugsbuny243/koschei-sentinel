from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.promotion import public_key_fingerprint
from koschei_sentinel.shadow_baseline import ShadowBaselineLineage, verify_shadow_baseline_lineage

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_ACTOR_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_CURRENT_SIGNATURE_CONTEXT = b"koschei-sentinel-owner-rotation-current-v1\0"
_NEXT_SIGNATURE_CONTEXT = b"koschei-sentinel-owner-rotation-next-v1\0"


class OwnerRotationBlocked(ValueError):
    """Raised when owner-key governance evidence fails closed."""


class OwnerRotationProposal(StrictModel):
    schema_version: Literal["sentinel.owner-key-rotation-proposal.v1"] = (
        "sentinel.owner-key-rotation-proposal.v1"
    )
    state: Literal["awaiting_current_owner_signature"] = "awaiting_current_owner_signature"
    authority: Literal["governance_evidence_only"] = "governance_evidence_only"
    lineage_digest: str = Field(pattern=_DIGEST)
    lineage_head_candidate_id: str = Field(pattern=_CANDIDATE_ID)
    replay_sha256: str = Field(pattern=_DIGEST)
    current_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    next_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    current_owner_signature_required: Literal[True] = True
    next_owner_counter_signature_required: Literal[True] = True
    baseline_rekey_required: Literal[True] = True
    automatic_key_activation_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    proposal_digest: str = Field(pattern=_DIGEST)


class CurrentOwnerRotationApproval(StrictModel):
    schema_version: Literal["sentinel.owner-key-rotation-current-approval.v1"] = (
        "sentinel.owner-key-rotation-current-approval.v1"
    )
    state: Literal["current_owner_signed_handoff"] = "current_owner_signed_handoff"
    authority: Literal["governance_evidence_only"] = "governance_evidence_only"
    approver_id: str = Field(pattern=_ACTOR_ID)
    current_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    next_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    automatic_key_activation_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    approval_digest: str = Field(pattern=_DIGEST)


class NextOwnerRotationAcceptance(StrictModel):
    schema_version: Literal["sentinel.owner-key-rotation-next-acceptance.v1"] = (
        "sentinel.owner-key-rotation-next-acceptance.v1"
    )
    state: Literal["next_owner_counter_signed_acceptance"] = (
        "next_owner_counter_signed_acceptance"
    )
    authority: Literal["governance_evidence_only"] = "governance_evidence_only"
    accepter_id: str = Field(pattern=_ACTOR_ID)
    current_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    next_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    current_approval_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    automatic_key_activation_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    acceptance_digest: str = Field(pattern=_DIGEST)


class OwnerRotationCheckpoint(StrictModel):
    schema_version: Literal["sentinel.owner-key-rotation-checkpoint.v1"] = (
        "sentinel.owner-key-rotation-checkpoint.v1"
    )
    state: Literal["dual_signed_not_activated"] = "dual_signed_not_activated"
    authority: Literal["governance_evidence_only"] = "governance_evidence_only"
    lineage_digest: str = Field(pattern=_DIGEST)
    lineage_head_candidate_id: str = Field(pattern=_CANDIDATE_ID)
    replay_sha256: str = Field(pattern=_DIGEST)
    current_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    next_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    current_approver_id: str = Field(pattern=_ACTOR_ID)
    current_owner_signature_base64: str = Field(min_length=80, max_length=128)
    current_approval_digest: str = Field(pattern=_DIGEST)
    next_accepter_id: str = Field(pattern=_ACTOR_ID)
    next_owner_signature_base64: str = Field(min_length=80, max_length=128)
    next_acceptance_digest: str = Field(pattern=_DIGEST)
    dual_signature_verified: Literal[True] = True
    baseline_rekey_required: Literal[True] = True
    automatic_key_activation_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    verdict_mutation_allowed: Literal[False] = False
    checkpoint_digest: str = Field(pattern=_DIGEST)


class OwnerRotationClaim(StrictModel):
    schema_version: Literal["sentinel.owner-key-rotation-claim.v1"] = (
        "sentinel.owner-key-rotation-claim.v1"
    )
    state: Literal["rotation_checkpoint_consumed"] = "rotation_checkpoint_consumed"
    authority: Literal["governance_evidence_only"] = "governance_evidence_only"
    lineage_digest: str = Field(pattern=_DIGEST)
    current_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    next_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    checkpoint_digest: str = Field(pattern=_DIGEST)
    automatic_key_activation_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    claim_digest: str = Field(pattern=_DIGEST)


def build_owner_rotation_proposal(
    lineage: ShadowBaselineLineage,
    current_owner_public_key: Ed25519PublicKey,
    next_owner_public_key: Ed25519PublicKey,
) -> OwnerRotationProposal:
    verify_shadow_baseline_lineage(lineage, current_owner_public_key)
    current = public_key_fingerprint(current_owner_public_key)
    next_fingerprint = public_key_fingerprint(next_owner_public_key)
    if current == next_fingerprint:
        raise OwnerRotationBlocked("next owner key must differ from current owner key")
    payload = _proposal_payload(
        lineage_digest=lineage.lineage_digest,
        lineage_head_candidate_id=lineage.head_candidate_id,
        replay_sha256=lineage.replay_sha256,
        current_owner_key_fingerprint=current,
        next_owner_key_fingerprint=next_fingerprint,
    )
    return OwnerRotationProposal.model_validate(
        {**payload, "proposal_digest": _digest(payload)}
    )


def approve_owner_rotation_proposal(
    proposal: OwnerRotationProposal,
    current_owner_private_key: Ed25519PrivateKey,
    *,
    approver_id: str,
) -> CurrentOwnerRotationApproval:
    _require_model_digest(proposal, "proposal_digest", "owner rotation proposal")
    fingerprint = public_key_fingerprint(current_owner_private_key.public_key())
    if fingerprint != proposal.current_owner_key_fingerprint:
        raise OwnerRotationBlocked("current owner private key does not match rotation proposal")
    signature = current_owner_private_key.sign(
        _current_signature_message(proposal.proposal_digest, approver_id)
    )
    payload = _current_approval_payload(
        approver_id=approver_id,
        current_owner_key_fingerprint=proposal.current_owner_key_fingerprint,
        next_owner_key_fingerprint=proposal.next_owner_key_fingerprint,
        proposal_digest=proposal.proposal_digest,
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )
    return CurrentOwnerRotationApproval.model_validate(
        {**payload, "approval_digest": _digest(payload)}
    )


def verify_current_owner_rotation_approval(
    proposal: OwnerRotationProposal,
    approval: CurrentOwnerRotationApproval,
    current_owner_public_key: Ed25519PublicKey,
) -> CurrentOwnerRotationApproval:
    _require_model_digest(proposal, "proposal_digest", "owner rotation proposal")
    _require_model_digest(approval, "approval_digest", "current owner rotation approval")
    fingerprint = public_key_fingerprint(current_owner_public_key)
    if fingerprint != proposal.current_owner_key_fingerprint:
        raise OwnerRotationBlocked("current public key does not match rotation proposal")
    if approval.current_owner_key_fingerprint != proposal.current_owner_key_fingerprint:
        raise OwnerRotationBlocked("current approval owner fingerprint does not match proposal")
    if approval.next_owner_key_fingerprint != proposal.next_owner_key_fingerprint:
        raise OwnerRotationBlocked(
            "current approval next owner fingerprint does not match proposal"
        )
    if approval.proposal_digest != proposal.proposal_digest:
        raise OwnerRotationBlocked("current approval does not bind rotation proposal")
    _verify_signature(
        current_owner_public_key,
        _current_signature_message(proposal.proposal_digest, approval.approver_id),
        approval.signature_base64,
        "current owner rotation signature verification failed",
    )
    return approval


def accept_owner_rotation(
    proposal: OwnerRotationProposal,
    current_approval: CurrentOwnerRotationApproval,
    current_owner_public_key: Ed25519PublicKey,
    next_owner_private_key: Ed25519PrivateKey,
    *,
    accepter_id: str,
) -> NextOwnerRotationAcceptance:
    verify_current_owner_rotation_approval(proposal, current_approval, current_owner_public_key)
    next_fingerprint = public_key_fingerprint(next_owner_private_key.public_key())
    if next_fingerprint != proposal.next_owner_key_fingerprint:
        raise OwnerRotationBlocked("next owner private key does not match rotation proposal")
    signature = next_owner_private_key.sign(
        _next_signature_message(
            proposal.proposal_digest,
            current_approval.approval_digest,
            accepter_id,
        )
    )
    payload = _next_acceptance_payload(
        accepter_id=accepter_id,
        current_owner_key_fingerprint=proposal.current_owner_key_fingerprint,
        next_owner_key_fingerprint=proposal.next_owner_key_fingerprint,
        proposal_digest=proposal.proposal_digest,
        current_approval_digest=current_approval.approval_digest,
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )
    return NextOwnerRotationAcceptance.model_validate(
        {**payload, "acceptance_digest": _digest(payload)}
    )


def verify_next_owner_rotation_acceptance(
    proposal: OwnerRotationProposal,
    current_approval: CurrentOwnerRotationApproval,
    acceptance: NextOwnerRotationAcceptance,
    current_owner_public_key: Ed25519PublicKey,
    next_owner_public_key: Ed25519PublicKey,
) -> NextOwnerRotationAcceptance:
    verify_current_owner_rotation_approval(proposal, current_approval, current_owner_public_key)
    _require_model_digest(acceptance, "acceptance_digest", "next owner rotation acceptance")
    next_fingerprint = public_key_fingerprint(next_owner_public_key)
    if next_fingerprint != proposal.next_owner_key_fingerprint:
        raise OwnerRotationBlocked("next public key does not match rotation proposal")
    if acceptance.current_owner_key_fingerprint != proposal.current_owner_key_fingerprint:
        raise OwnerRotationBlocked("next acceptance current owner does not match proposal")
    if acceptance.next_owner_key_fingerprint != proposal.next_owner_key_fingerprint:
        raise OwnerRotationBlocked("next acceptance owner fingerprint does not match proposal")
    if acceptance.proposal_digest != proposal.proposal_digest:
        raise OwnerRotationBlocked("next acceptance does not bind rotation proposal")
    if acceptance.current_approval_digest != current_approval.approval_digest:
        raise OwnerRotationBlocked("next acceptance does not bind current owner approval")
    _verify_signature(
        next_owner_public_key,
        _next_signature_message(
            proposal.proposal_digest,
            current_approval.approval_digest,
            acceptance.accepter_id,
        ),
        acceptance.signature_base64,
        "next owner rotation signature verification failed",
    )
    return acceptance


def build_owner_rotation_checkpoint(
    proposal: OwnerRotationProposal,
    current_approval: CurrentOwnerRotationApproval,
    acceptance: NextOwnerRotationAcceptance,
    lineage: ShadowBaselineLineage,
    current_owner_public_key: Ed25519PublicKey,
    next_owner_public_key: Ed25519PublicKey,
) -> OwnerRotationCheckpoint:
    expected = build_owner_rotation_proposal(
        lineage,
        current_owner_public_key,
        next_owner_public_key,
    )
    if expected.model_dump(mode="json") != proposal.model_dump(mode="json"):
        raise OwnerRotationBlocked("rotation proposal does not match supplied lineage and keys")
    verify_next_owner_rotation_acceptance(
        proposal,
        current_approval,
        acceptance,
        current_owner_public_key,
        next_owner_public_key,
    )
    payload = _checkpoint_payload(
        lineage_digest=lineage.lineage_digest,
        lineage_head_candidate_id=lineage.head_candidate_id,
        replay_sha256=lineage.replay_sha256,
        current_owner_key_fingerprint=proposal.current_owner_key_fingerprint,
        next_owner_key_fingerprint=proposal.next_owner_key_fingerprint,
        proposal_digest=proposal.proposal_digest,
        current_approver_id=current_approval.approver_id,
        current_owner_signature_base64=current_approval.signature_base64,
        current_approval_digest=current_approval.approval_digest,
        next_accepter_id=acceptance.accepter_id,
        next_owner_signature_base64=acceptance.signature_base64,
        next_acceptance_digest=acceptance.acceptance_digest,
    )
    checkpoint = OwnerRotationCheckpoint.model_validate(
        {**payload, "checkpoint_digest": _digest(payload)}
    )
    verify_owner_rotation_checkpoint(
        checkpoint,
        lineage,
        current_owner_public_key,
        next_owner_public_key,
    )
    return checkpoint


def verify_owner_rotation_checkpoint(
    checkpoint: OwnerRotationCheckpoint,
    lineage: ShadowBaselineLineage,
    current_owner_public_key: Ed25519PublicKey,
    next_owner_public_key: Ed25519PublicKey,
) -> OwnerRotationCheckpoint:
    _require_model_digest(checkpoint, "checkpoint_digest", "owner rotation checkpoint")
    proposal = build_owner_rotation_proposal(
        lineage,
        current_owner_public_key,
        next_owner_public_key,
    )
    if checkpoint.lineage_digest != proposal.lineage_digest:
        raise OwnerRotationBlocked("checkpoint lineage digest does not match current baseline")
    if checkpoint.lineage_head_candidate_id != proposal.lineage_head_candidate_id:
        raise OwnerRotationBlocked("checkpoint lineage head does not match current baseline")
    if checkpoint.replay_sha256 != proposal.replay_sha256:
        raise OwnerRotationBlocked("checkpoint replay does not match current baseline")
    if checkpoint.current_owner_key_fingerprint != proposal.current_owner_key_fingerprint:
        raise OwnerRotationBlocked("checkpoint current owner key does not match proposal")
    if checkpoint.next_owner_key_fingerprint != proposal.next_owner_key_fingerprint:
        raise OwnerRotationBlocked("checkpoint next owner key does not match proposal")
    if checkpoint.proposal_digest != proposal.proposal_digest:
        raise OwnerRotationBlocked("checkpoint proposal digest is inconsistent")

    current_payload = _current_approval_payload(
        approver_id=checkpoint.current_approver_id,
        current_owner_key_fingerprint=checkpoint.current_owner_key_fingerprint,
        next_owner_key_fingerprint=checkpoint.next_owner_key_fingerprint,
        proposal_digest=checkpoint.proposal_digest,
        signature_base64=checkpoint.current_owner_signature_base64,
    )
    current_approval = CurrentOwnerRotationApproval.model_validate(
        {**current_payload, "approval_digest": checkpoint.current_approval_digest}
    )
    verify_current_owner_rotation_approval(proposal, current_approval, current_owner_public_key)

    next_payload = _next_acceptance_payload(
        accepter_id=checkpoint.next_accepter_id,
        current_owner_key_fingerprint=checkpoint.current_owner_key_fingerprint,
        next_owner_key_fingerprint=checkpoint.next_owner_key_fingerprint,
        proposal_digest=checkpoint.proposal_digest,
        current_approval_digest=checkpoint.current_approval_digest,
        signature_base64=checkpoint.next_owner_signature_base64,
    )
    acceptance = NextOwnerRotationAcceptance.model_validate(
        {**next_payload, "acceptance_digest": checkpoint.next_acceptance_digest}
    )
    verify_next_owner_rotation_acceptance(
        proposal,
        current_approval,
        acceptance,
        current_owner_public_key,
        next_owner_public_key,
    )
    return checkpoint


def build_owner_rotation_claim(checkpoint: OwnerRotationCheckpoint) -> OwnerRotationClaim:
    _require_model_digest(checkpoint, "checkpoint_digest", "owner rotation checkpoint")
    payload = {
        "schema_version": "sentinel.owner-key-rotation-claim.v1",
        "state": "rotation_checkpoint_consumed",
        "authority": "governance_evidence_only",
        "lineage_digest": checkpoint.lineage_digest,
        "current_owner_key_fingerprint": checkpoint.current_owner_key_fingerprint,
        "next_owner_key_fingerprint": checkpoint.next_owner_key_fingerprint,
        "checkpoint_digest": checkpoint.checkpoint_digest,
        "automatic_key_activation_allowed": False,
        "production_deployment_allowed": False,
    }
    return OwnerRotationClaim.model_validate({**payload, "claim_digest": _digest(payload)})


def claim_owner_rotation(checkpoint: OwnerRotationCheckpoint, claim_dir: str | Path) -> Path:
    claim = build_owner_rotation_claim(checkpoint)
    root = Path(claim_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (
        f"{claim.current_owner_key_fingerprint}.{claim.lineage_digest}.json"
    )
    serialized = json.dumps(claim.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if destination.exists():
        _accept_same_or_reject_rotation_fork(destination, claim)
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=root,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            _accept_same_or_reject_rotation_fork(destination, claim)
        return destination
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_owner_rotation_proposal(path: str | Path) -> OwnerRotationProposal:
    return _load_model(path, OwnerRotationProposal, "owner rotation proposal", "proposal_digest")


def load_current_owner_rotation_approval(path: str | Path) -> CurrentOwnerRotationApproval:
    return _load_model(
        path,
        CurrentOwnerRotationApproval,
        "current owner rotation approval",
        "approval_digest",
    )


def load_next_owner_rotation_acceptance(path: str | Path) -> NextOwnerRotationAcceptance:
    return _load_model(
        path,
        NextOwnerRotationAcceptance,
        "next owner rotation acceptance",
        "acceptance_digest",
    )


def load_owner_rotation_checkpoint(path: str | Path) -> OwnerRotationCheckpoint:
    return _load_model(
        path,
        OwnerRotationCheckpoint,
        "owner rotation checkpoint",
        "checkpoint_digest",
    )


def write_owner_rotation_artifact(model: StrictModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"owner rotation artifact already exists: {destination}") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _proposal_payload(
    *,
    lineage_digest: str,
    lineage_head_candidate_id: str,
    replay_sha256: str,
    current_owner_key_fingerprint: str,
    next_owner_key_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.owner-key-rotation-proposal.v1",
        "state": "awaiting_current_owner_signature",
        "authority": "governance_evidence_only",
        "lineage_digest": lineage_digest,
        "lineage_head_candidate_id": lineage_head_candidate_id,
        "replay_sha256": replay_sha256,
        "current_owner_key_fingerprint": current_owner_key_fingerprint,
        "next_owner_key_fingerprint": next_owner_key_fingerprint,
        "current_owner_signature_required": True,
        "next_owner_counter_signature_required": True,
        "baseline_rekey_required": True,
        "automatic_key_activation_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }


def _current_approval_payload(
    *,
    approver_id: str,
    current_owner_key_fingerprint: str,
    next_owner_key_fingerprint: str,
    proposal_digest: str,
    signature_base64: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.owner-key-rotation-current-approval.v1",
        "state": "current_owner_signed_handoff",
        "authority": "governance_evidence_only",
        "approver_id": approver_id,
        "current_owner_key_fingerprint": current_owner_key_fingerprint,
        "next_owner_key_fingerprint": next_owner_key_fingerprint,
        "proposal_digest": proposal_digest,
        "signature_algorithm": "ed25519",
        "signature_base64": signature_base64,
        "signature_verified": True,
        "automatic_key_activation_allowed": False,
        "production_deployment_allowed": False,
    }


def _next_acceptance_payload(
    *,
    accepter_id: str,
    current_owner_key_fingerprint: str,
    next_owner_key_fingerprint: str,
    proposal_digest: str,
    current_approval_digest: str,
    signature_base64: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.owner-key-rotation-next-acceptance.v1",
        "state": "next_owner_counter_signed_acceptance",
        "authority": "governance_evidence_only",
        "accepter_id": accepter_id,
        "current_owner_key_fingerprint": current_owner_key_fingerprint,
        "next_owner_key_fingerprint": next_owner_key_fingerprint,
        "proposal_digest": proposal_digest,
        "current_approval_digest": current_approval_digest,
        "signature_algorithm": "ed25519",
        "signature_base64": signature_base64,
        "signature_verified": True,
        "automatic_key_activation_allowed": False,
        "production_deployment_allowed": False,
    }


def _checkpoint_payload(
    *,
    lineage_digest: str,
    lineage_head_candidate_id: str,
    replay_sha256: str,
    current_owner_key_fingerprint: str,
    next_owner_key_fingerprint: str,
    proposal_digest: str,
    current_approver_id: str,
    current_owner_signature_base64: str,
    current_approval_digest: str,
    next_accepter_id: str,
    next_owner_signature_base64: str,
    next_acceptance_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.owner-key-rotation-checkpoint.v1",
        "state": "dual_signed_not_activated",
        "authority": "governance_evidence_only",
        "lineage_digest": lineage_digest,
        "lineage_head_candidate_id": lineage_head_candidate_id,
        "replay_sha256": replay_sha256,
        "current_owner_key_fingerprint": current_owner_key_fingerprint,
        "next_owner_key_fingerprint": next_owner_key_fingerprint,
        "proposal_digest": proposal_digest,
        "current_approver_id": current_approver_id,
        "current_owner_signature_base64": current_owner_signature_base64,
        "current_approval_digest": current_approval_digest,
        "next_accepter_id": next_accepter_id,
        "next_owner_signature_base64": next_owner_signature_base64,
        "next_acceptance_digest": next_acceptance_digest,
        "dual_signature_verified": True,
        "baseline_rekey_required": True,
        "automatic_key_activation_allowed": False,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }


def _current_signature_message(proposal_digest: str, approver_id: str) -> bytes:
    return (
        _CURRENT_SIGNATURE_CONTEXT
        + proposal_digest.encode("ascii")
        + b"\0"
        + approver_id.encode()
    )


def _next_signature_message(
    proposal_digest: str,
    current_approval_digest: str,
    accepter_id: str,
) -> bytes:
    return (
        _NEXT_SIGNATURE_CONTEXT
        + proposal_digest.encode("ascii")
        + b"\0"
        + current_approval_digest.encode("ascii")
        + b"\0"
        + accepter_id.encode()
    )


def _verify_signature(
    public_key: Ed25519PublicKey,
    message: bytes,
    signature_base64: str,
    error_message: str,
) -> None:
    try:
        public_key.verify(base64.b64decode(signature_base64, validate=True), message)
    except (InvalidSignature, ValueError) as exc:
        raise OwnerRotationBlocked(error_message) from exc


def _load_model(
    path: str | Path,
    model_type: type[StrictModel],
    label: str,
    digest_field: str,
):
    try:
        model = model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc
    _require_model_digest(model, digest_field, label)
    return model


def _accept_same_or_reject_rotation_fork(
    destination: Path,
    expected: OwnerRotationClaim,
) -> None:
    try:
        existing = OwnerRotationClaim.model_validate_json(
            destination.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise OwnerRotationBlocked("existing owner rotation claim is invalid") from exc
    _require_model_digest(existing, "claim_digest", "owner rotation claim")
    if existing.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise OwnerRotationBlocked(
            "baseline checkpoint already has a different owner rotation claim"
        )


def _require_model_digest(model: StrictModel, field: str, label: str) -> None:
    payload = model.model_dump(mode="json")
    claimed = payload.pop(field)
    if claimed != _digest(payload):
        raise OwnerRotationBlocked(f"{label} digest does not match contents")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CurrentOwnerRotationApproval",
    "NextOwnerRotationAcceptance",
    "OwnerRotationBlocked",
    "OwnerRotationCheckpoint",
    "OwnerRotationClaim",
    "OwnerRotationProposal",
    "accept_owner_rotation",
    "approve_owner_rotation_proposal",
    "build_owner_rotation_checkpoint",
    "build_owner_rotation_claim",
    "build_owner_rotation_proposal",
    "claim_owner_rotation",
    "load_current_owner_rotation_approval",
    "load_next_owner_rotation_acceptance",
    "load_owner_rotation_checkpoint",
    "load_owner_rotation_proposal",
    "verify_current_owner_rotation_approval",
    "verify_next_owner_rotation_acceptance",
    "verify_owner_rotation_checkpoint",
    "write_owner_rotation_artifact",
]
