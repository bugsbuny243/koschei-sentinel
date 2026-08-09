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
from koschei_sentinel.shadow_receipt import ShadowReplayReceipt
from koschei_sentinel.shadow_regression import ShadowRegressionReport
from koschei_sentinel.shadow_review import ShadowReviewScorecard

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_APPROVER_ID = r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$"
_SIGNATURE_CONTEXT = b"koschei-sentinel-shadow-baseline-v1\0"
_EMPTY_DIGEST = "0" * 64
_MAX_LINEAGE = 10_000


class ShadowBaselineBlocked(ValueError):
    """Raised when a shadow baseline lineage operation fails closed."""


class ShadowBaselineProposal(StrictModel):
    schema_version: Literal["sentinel.shadow-baseline-proposal.v1"] = (
        "sentinel.shadow-baseline-proposal.v1"
    )
    state: Literal["awaiting_owner_signature"] = "awaiting_owner_signature"
    authority: Literal["explanation_only"] = "explanation_only"
    baseline_candidate_id: str = Field(pattern=_CANDIDATE_ID)
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    replay_sha256: str = Field(pattern=_DIGEST)
    regression_report_digest: str = Field(pattern=_DIGEST)
    baseline_scorecard_digest: str = Field(pattern=_DIGEST)
    baseline_receipt_digest: str = Field(pattern=_DIGEST)
    candidate_scorecard_digest: str = Field(pattern=_DIGEST)
    candidate_receipt_digest: str = Field(pattern=_DIGEST)
    previous_lineage_digest: str = Field(pattern=_DIGEST)
    previous_head_candidate_id: str | None = Field(default=None, pattern=_CANDIDATE_ID)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    owner_signature_required: Literal[True] = True
    manual_review_required: Literal[True] = True
    automatic_baseline_selection_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    verdict_mutation_allowed: Literal[False] = False
    proposal_digest: str = Field(pattern=_DIGEST)


class ShadowBaselineApproval(StrictModel):
    schema_version: Literal["sentinel.shadow-baseline-approval.v1"] = (
        "sentinel.shadow-baseline-approval.v1"
    )
    state: Literal["owner_approved_baseline_advance"] = "owner_approved_baseline_advance"
    authority: Literal["explanation_only"] = "explanation_only"
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    approver_id: str = Field(pattern=_APPROVER_ID)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    signature_algorithm: Literal["ed25519"] = "ed25519"
    signature_base64: str = Field(min_length=80, max_length=128)
    signature_verified: Literal[True] = True
    automatic_baseline_selection_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    approval_digest: str = Field(pattern=_DIGEST)


class ShadowBaselineEntry(StrictModel):
    ordinal: int = Field(ge=0)
    kind: Literal["seed", "advance"]
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    parent_candidate_id: str | None = Field(default=None, pattern=_CANDIDATE_ID)
    scorecard_digest: str = Field(pattern=_DIGEST)
    receipt_digest: str = Field(pattern=_DIGEST)
    replay_sha256: str = Field(pattern=_DIGEST)
    regression_report_digest: str = Field(pattern=_DIGEST)
    owner_approval_digest: str = Field(pattern=_DIGEST)


class ShadowBaselineLineage(StrictModel):
    schema_version: Literal["sentinel.shadow-baseline-lineage.v1"] = (
        "sentinel.shadow-baseline-lineage.v1"
    )
    state: Literal["owner_curated_shadow_baseline_lineage"] = (
        "owner_curated_shadow_baseline_lineage"
    )
    authority: Literal["explanation_only"] = "explanation_only"
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    replay_sha256: str = Field(pattern=_DIGEST)
    head_candidate_id: str = Field(pattern=_CANDIDATE_ID)
    head_scorecard_digest: str = Field(pattern=_DIGEST)
    head_receipt_digest: str = Field(pattern=_DIGEST)
    entries: list[ShadowBaselineEntry] = Field(min_length=1, max_length=_MAX_LINEAGE)
    owner_signature_required_for_advance: Literal[True] = True
    manual_review_required: Literal[True] = True
    automatic_baseline_selection_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    verdict_mutation_allowed: Literal[False] = False
    lineage_digest: str = Field(pattern=_DIGEST)


def build_shadow_baseline_proposal(
    report: ShadowRegressionReport,
    baseline_scorecard: ShadowReviewScorecard,
    baseline_receipt: ShadowReplayReceipt,
    candidate_scorecard: ShadowReviewScorecard,
    candidate_receipt: ShadowReplayReceipt,
    owner_public_key: Ed25519PublicKey,
    *,
    lineage: ShadowBaselineLineage | None = None,
) -> ShadowBaselineProposal:
    _verify_evidence_bundle(
        report,
        baseline_scorecard,
        baseline_receipt,
        candidate_scorecard,
        candidate_receipt,
    )
    if not report.regression_passed:
        raise ShadowBaselineBlocked("candidate did not pass the shadow regression gate")
    fingerprint = public_key_fingerprint(owner_public_key)
    if lineage is None:
        previous_digest = _EMPTY_DIGEST
        previous_head = None
    else:
        verify_shadow_baseline_lineage(lineage)
        if lineage.owner_key_fingerprint != fingerprint:
            raise ShadowBaselineBlocked("owner key does not match existing baseline lineage")
        if lineage.replay_sha256 != report.replay_sha256:
            raise ShadowBaselineBlocked("baseline lineage uses a different sealed replay")
        if lineage.head_candidate_id != report.baseline_candidate_id:
            raise ShadowBaselineBlocked("regression baseline is not the current lineage head")
        if lineage.head_scorecard_digest != report.baseline_scorecard_digest:
            raise ShadowBaselineBlocked("lineage head scorecard does not match regression baseline")
        if lineage.head_receipt_digest != report.baseline_receipt_digest:
            raise ShadowBaselineBlocked("lineage head receipt does not match regression baseline")
        if any(entry.candidate_id == report.candidate_id for entry in lineage.entries):
            raise ShadowBaselineBlocked("candidate already exists in baseline lineage")
        previous_digest = lineage.lineage_digest
        previous_head = lineage.head_candidate_id

    payload = {
        "schema_version": "sentinel.shadow-baseline-proposal.v1",
        "state": "awaiting_owner_signature",
        "authority": "explanation_only",
        "baseline_candidate_id": report.baseline_candidate_id,
        "candidate_id": report.candidate_id,
        "replay_sha256": report.replay_sha256,
        "regression_report_digest": report.report_digest,
        "baseline_scorecard_digest": report.baseline_scorecard_digest,
        "baseline_receipt_digest": report.baseline_receipt_digest,
        "candidate_scorecard_digest": report.candidate_scorecard_digest,
        "candidate_receipt_digest": report.candidate_receipt_digest,
        "previous_lineage_digest": previous_digest,
        "previous_head_candidate_id": previous_head,
        "owner_key_fingerprint": fingerprint,
        "owner_signature_required": True,
        "manual_review_required": True,
        "automatic_baseline_selection_allowed": False,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowBaselineProposal.model_validate(
        {**payload, "proposal_digest": _digest(payload)}
    )


def approve_shadow_baseline_proposal(
    proposal: ShadowBaselineProposal,
    owner_private_key: Ed25519PrivateKey,
    *,
    approver_id: str,
) -> ShadowBaselineApproval:
    _require_model_digest(proposal, "proposal_digest", "baseline proposal")
    fingerprint = public_key_fingerprint(owner_private_key.public_key())
    if fingerprint != proposal.owner_key_fingerprint:
        raise ShadowBaselineBlocked("private key does not match baseline proposal owner key")
    signature = owner_private_key.sign(_signature_message(proposal.proposal_digest))
    payload = {
        "schema_version": "sentinel.shadow-baseline-approval.v1",
        "state": "owner_approved_baseline_advance",
        "authority": "explanation_only",
        "candidate_id": proposal.candidate_id,
        "approver_id": approver_id,
        "owner_key_fingerprint": fingerprint,
        "proposal_digest": proposal.proposal_digest,
        "signature_algorithm": "ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signature_verified": True,
        "automatic_baseline_selection_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return ShadowBaselineApproval.model_validate(
        {**payload, "approval_digest": _digest(payload)}
    )


def verify_shadow_baseline_approval(
    proposal: ShadowBaselineProposal,
    approval: ShadowBaselineApproval,
    owner_public_key: Ed25519PublicKey,
) -> ShadowBaselineApproval:
    _require_model_digest(proposal, "proposal_digest", "baseline proposal")
    _require_model_digest(approval, "approval_digest", "baseline approval")
    fingerprint = public_key_fingerprint(owner_public_key)
    if proposal.owner_key_fingerprint != fingerprint:
        raise ShadowBaselineBlocked("public key does not match baseline proposal")
    if approval.owner_key_fingerprint != fingerprint:
        raise ShadowBaselineBlocked("public key does not match baseline approval")
    if approval.candidate_id != proposal.candidate_id:
        raise ShadowBaselineBlocked("approval candidate does not match proposal")
    if approval.proposal_digest != proposal.proposal_digest:
        raise ShadowBaselineBlocked("approval does not bind supplied baseline proposal")
    try:
        signature = base64.b64decode(approval.signature_base64, validate=True)
        owner_public_key.verify(signature, _signature_message(proposal.proposal_digest))
    except (InvalidSignature, ValueError) as exc:
        raise ShadowBaselineBlocked("baseline owner signature verification failed") from exc
    return approval


def apply_shadow_baseline_advance(
    proposal: ShadowBaselineProposal,
    approval: ShadowBaselineApproval,
    report: ShadowRegressionReport,
    baseline_scorecard: ShadowReviewScorecard,
    baseline_receipt: ShadowReplayReceipt,
    candidate_scorecard: ShadowReviewScorecard,
    candidate_receipt: ShadowReplayReceipt,
    owner_public_key: Ed25519PublicKey,
    *,
    lineage: ShadowBaselineLineage | None = None,
) -> ShadowBaselineLineage:
    expected = build_shadow_baseline_proposal(
        report,
        baseline_scorecard,
        baseline_receipt,
        candidate_scorecard,
        candidate_receipt,
        owner_public_key,
        lineage=lineage,
    )
    if expected.model_dump(mode="json") != proposal.model_dump(mode="json"):
        raise ShadowBaselineBlocked("baseline proposal does not match supplied verified evidence")
    verify_shadow_baseline_approval(proposal, approval, owner_public_key)

    if lineage is None:
        entries = [
            ShadowBaselineEntry(
                ordinal=0,
                kind="seed",
                candidate_id=report.baseline_candidate_id,
                parent_candidate_id=None,
                scorecard_digest=report.baseline_scorecard_digest,
                receipt_digest=report.baseline_receipt_digest,
                replay_sha256=report.replay_sha256,
                regression_report_digest=report.report_digest,
                owner_approval_digest=approval.approval_digest,
            ),
            ShadowBaselineEntry(
                ordinal=1,
                kind="advance",
                candidate_id=report.candidate_id,
                parent_candidate_id=report.baseline_candidate_id,
                scorecard_digest=report.candidate_scorecard_digest,
                receipt_digest=report.candidate_receipt_digest,
                replay_sha256=report.replay_sha256,
                regression_report_digest=report.report_digest,
                owner_approval_digest=approval.approval_digest,
            ),
        ]
    else:
        verify_shadow_baseline_lineage(lineage)
        entries = [
            *lineage.entries,
            ShadowBaselineEntry(
                ordinal=len(lineage.entries),
                kind="advance",
                candidate_id=report.candidate_id,
                parent_candidate_id=lineage.head_candidate_id,
                scorecard_digest=report.candidate_scorecard_digest,
                receipt_digest=report.candidate_receipt_digest,
                replay_sha256=report.replay_sha256,
                regression_report_digest=report.report_digest,
                owner_approval_digest=approval.approval_digest,
            ),
        ]
    if len(entries) > _MAX_LINEAGE:
        raise ShadowBaselineBlocked("baseline lineage exceeds its entry limit")

    payload = {
        "schema_version": "sentinel.shadow-baseline-lineage.v1",
        "state": "owner_curated_shadow_baseline_lineage",
        "authority": "explanation_only",
        "owner_key_fingerprint": proposal.owner_key_fingerprint,
        "replay_sha256": report.replay_sha256,
        "head_candidate_id": report.candidate_id,
        "head_scorecard_digest": report.candidate_scorecard_digest,
        "head_receipt_digest": report.candidate_receipt_digest,
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "owner_signature_required_for_advance": True,
        "manual_review_required": True,
        "automatic_baseline_selection_allowed": False,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    result = ShadowBaselineLineage.model_validate(
        {**payload, "lineage_digest": _digest(payload)}
    )
    verify_shadow_baseline_lineage(result)
    return result


def verify_shadow_baseline_lineage(lineage: ShadowBaselineLineage) -> ShadowBaselineLineage:
    _require_model_digest(lineage, "lineage_digest", "baseline lineage")
    if not lineage.entries:
        raise ShadowBaselineBlocked("baseline lineage must not be empty")
    seen: set[str] = set()
    previous: ShadowBaselineEntry | None = None
    for ordinal, entry in enumerate(lineage.entries):
        if entry.ordinal != ordinal:
            raise ShadowBaselineBlocked("baseline lineage ordinals are not contiguous")
        if entry.replay_sha256 != lineage.replay_sha256:
            raise ShadowBaselineBlocked("baseline lineage mixes sealed replay datasets")
        if entry.candidate_id in seen:
            raise ShadowBaselineBlocked("baseline lineage contains a candidate cycle")
        seen.add(entry.candidate_id)
        if ordinal == 0:
            if entry.kind != "seed" or entry.parent_candidate_id is not None:
                raise ShadowBaselineBlocked("baseline lineage root is invalid")
        else:
            if entry.kind != "advance" or previous is None:
                raise ShadowBaselineBlocked("baseline lineage advance entry is invalid")
            if entry.parent_candidate_id != previous.candidate_id:
                raise ShadowBaselineBlocked("baseline lineage parent chain is broken")
        previous = entry
    head = lineage.entries[-1]
    if lineage.head_candidate_id != head.candidate_id:
        raise ShadowBaselineBlocked("baseline lineage head candidate is inconsistent")
    if lineage.head_scorecard_digest != head.scorecard_digest:
        raise ShadowBaselineBlocked("baseline lineage head scorecard is inconsistent")
    if lineage.head_receipt_digest != head.receipt_digest:
        raise ShadowBaselineBlocked("baseline lineage head receipt is inconsistent")
    return lineage


def load_shadow_baseline_proposal(path: str | Path) -> ShadowBaselineProposal:
    proposal = _load_model(path, ShadowBaselineProposal, "baseline proposal")
    _require_model_digest(proposal, "proposal_digest", "baseline proposal")
    return proposal


def load_shadow_baseline_approval(path: str | Path) -> ShadowBaselineApproval:
    approval = _load_model(path, ShadowBaselineApproval, "baseline approval")
    _require_model_digest(approval, "approval_digest", "baseline approval")
    return approval


def load_shadow_baseline_lineage(path: str | Path) -> ShadowBaselineLineage:
    lineage = _load_model(path, ShadowBaselineLineage, "baseline lineage")
    return verify_shadow_baseline_lineage(lineage)


def write_shadow_baseline_artifact(model: StrictModel, path: str | Path) -> None:
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
            raise FileExistsError(f"baseline artifact already exists: {destination}") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _verify_evidence_bundle(
    report: ShadowRegressionReport,
    baseline_scorecard: ShadowReviewScorecard,
    baseline_receipt: ShadowReplayReceipt,
    candidate_scorecard: ShadowReviewScorecard,
    candidate_receipt: ShadowReplayReceipt,
) -> None:
    _require_model_digest(report, "report_digest", "shadow regression report")
    _require_model_digest(baseline_scorecard, "scorecard_digest", "baseline scorecard")
    _require_model_digest(candidate_scorecard, "scorecard_digest", "candidate scorecard")
    _require_model_digest(baseline_receipt, "receipt_digest", "baseline receipt")
    _require_model_digest(candidate_receipt, "receipt_digest", "candidate receipt")
    if not baseline_scorecard.gate_passed or not candidate_scorecard.gate_passed:
        raise ShadowBaselineBlocked("baseline lineage requires passing human review scorecards")
    if report.baseline_candidate_id != baseline_scorecard.candidate_id:
        raise ShadowBaselineBlocked("regression baseline candidate does not match scorecard")
    if report.candidate_id != candidate_scorecard.candidate_id:
        raise ShadowBaselineBlocked("regression candidate does not match scorecard")
    bindings = (
        (
            report.baseline_scorecard_digest,
            baseline_scorecard.scorecard_digest,
            "baseline scorecard",
        ),
        (
            report.candidate_scorecard_digest,
            candidate_scorecard.scorecard_digest,
            "candidate scorecard",
        ),
        (report.baseline_receipt_digest, baseline_receipt.receipt_digest, "baseline receipt"),
        (report.candidate_receipt_digest, candidate_receipt.receipt_digest, "candidate receipt"),
    )
    for claimed, observed, label in bindings:
        if claimed != observed:
            raise ShadowBaselineBlocked(f"regression report does not bind supplied {label}")
    if baseline_scorecard.receipt_digest != baseline_receipt.receipt_digest:
        raise ShadowBaselineBlocked("baseline scorecard does not bind supplied receipt")
    if candidate_scorecard.receipt_digest != candidate_receipt.receipt_digest:
        raise ShadowBaselineBlocked("candidate scorecard does not bind supplied receipt")
    if baseline_receipt.replay_sha256 != report.replay_sha256:
        raise ShadowBaselineBlocked("baseline receipt uses a different sealed replay")
    if candidate_receipt.replay_sha256 != report.replay_sha256:
        raise ShadowBaselineBlocked("candidate receipt uses a different sealed replay")


def _load_model(path: str | Path, model_type: type[StrictModel], label: str) -> StrictModel:
    try:
        return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc


def _require_model_digest(model: StrictModel, field: str, label: str) -> None:
    payload = model.model_dump(mode="json")
    claimed = payload.pop(field)
    if claimed != _digest(payload):
        raise ShadowBaselineBlocked(f"{label} digest does not match contents")


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


__all__ = [
    "ShadowBaselineApproval",
    "ShadowBaselineBlocked",
    "ShadowBaselineEntry",
    "ShadowBaselineLineage",
    "ShadowBaselineProposal",
    "apply_shadow_baseline_advance",
    "approve_shadow_baseline_proposal",
    "build_shadow_baseline_proposal",
    "load_shadow_baseline_approval",
    "load_shadow_baseline_lineage",
    "load_shadow_baseline_proposal",
    "verify_shadow_baseline_approval",
    "verify_shadow_baseline_lineage",
    "write_shadow_baseline_artifact",
]
