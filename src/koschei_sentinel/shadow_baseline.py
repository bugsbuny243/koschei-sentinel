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
from koschei_sentinel.shadow_regression import (
    ShadowRegressionReport,
    build_shadow_regression_report,
)
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
    proposal_digest: str | None = Field(default=None, pattern=_DIGEST)
    approver_id: str | None = Field(default=None, pattern=_APPROVER_ID)
    owner_signature_base64: str | None = Field(default=None, min_length=80, max_length=128)
    owner_approval_digest: str | None = Field(default=None, pattern=_DIGEST)


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
    entries: list[ShadowBaselineEntry] = Field(min_length=2, max_length=_MAX_LINEAGE)
    owner_signature_required_for_advance: Literal[True] = True
    historical_signatures_verified: Literal[True] = True
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
        verify_shadow_baseline_lineage(lineage, owner_public_key)
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

    payload = _proposal_payload(
        baseline_candidate_id=report.baseline_candidate_id,
        candidate_id=report.candidate_id,
        replay_sha256=report.replay_sha256,
        regression_report_digest=report.report_digest,
        baseline_scorecard_digest=report.baseline_scorecard_digest,
        baseline_receipt_digest=report.baseline_receipt_digest,
        candidate_scorecard_digest=report.candidate_scorecard_digest,
        candidate_receipt_digest=report.candidate_receipt_digest,
        previous_lineage_digest=previous_digest,
        previous_head_candidate_id=previous_head,
        owner_key_fingerprint=fingerprint,
    )
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
    signature = owner_private_key.sign(
        _signature_message(proposal.proposal_digest, approver_id)
    )
    payload = _approval_payload(
        candidate_id=proposal.candidate_id,
        approver_id=approver_id,
        owner_key_fingerprint=fingerprint,
        proposal_digest=proposal.proposal_digest,
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )
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
    _verify_signature(
        owner_public_key,
        proposal.proposal_digest,
        approval.approver_id,
        approval.signature_base64,
    )
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

    advance = _advance_entry(proposal, approval)
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
                regression_report_digest=_EMPTY_DIGEST,
            ),
            advance,
        ]
    else:
        verify_shadow_baseline_lineage(lineage, owner_public_key)
        entries = [
            *lineage.entries,
            advance.model_copy(update={"ordinal": len(lineage.entries)}),
        ]
    if len(entries) > _MAX_LINEAGE:
        raise ShadowBaselineBlocked("baseline lineage exceeds its entry limit")

    result = _make_lineage(
        owner_key_fingerprint=proposal.owner_key_fingerprint,
        replay_sha256=report.replay_sha256,
        entries=entries,
    )
    verify_shadow_baseline_lineage(result, owner_public_key)
    return result


def verify_shadow_baseline_lineage(
    lineage: ShadowBaselineLineage,
    owner_public_key: Ed25519PublicKey,
) -> ShadowBaselineLineage:
    _require_model_digest(lineage, "lineage_digest", "baseline lineage")
    fingerprint = public_key_fingerprint(owner_public_key)
    if lineage.owner_key_fingerprint != fingerprint:
        raise ShadowBaselineBlocked("public key does not match baseline lineage owner")

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
            _verify_seed_entry(entry)
        else:
            if previous is None:
                raise ShadowBaselineBlocked("baseline lineage advance has no parent")
            _verify_advance_entry(
                lineage,
                entry,
                previous,
                owner_public_key,
                prefix_entries=lineage.entries[:ordinal],
            )
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
    _require_model_digest(lineage, "lineage_digest", "baseline lineage")
    return lineage


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

    expected_report = build_shadow_regression_report(
        baseline_scorecard,
        baseline_receipt,
        candidate_scorecard,
        candidate_receipt,
        tolerance=report.tolerance,
    )
    if expected_report.model_dump(mode="json") != report.model_dump(mode="json"):
        raise ShadowBaselineBlocked(
            "shadow regression report does not match supplied scorecards and receipts"
        )


def _verify_seed_entry(entry: ShadowBaselineEntry) -> None:
    if entry.kind != "seed" or entry.parent_candidate_id is not None:
        raise ShadowBaselineBlocked("baseline lineage root is invalid")
    if entry.regression_report_digest != _EMPTY_DIGEST:
        raise ShadowBaselineBlocked("baseline lineage seed must not claim a regression")
    if any(
        value is not None
        for value in (
            entry.proposal_digest,
            entry.approver_id,
            entry.owner_signature_base64,
            entry.owner_approval_digest,
        )
    ):
        raise ShadowBaselineBlocked("baseline lineage seed must not claim owner approval")


def _verify_advance_entry(
    lineage: ShadowBaselineLineage,
    entry: ShadowBaselineEntry,
    previous: ShadowBaselineEntry,
    owner_public_key: Ed25519PublicKey,
    *,
    prefix_entries: list[ShadowBaselineEntry],
) -> None:
    if entry.kind != "advance" or entry.parent_candidate_id != previous.candidate_id:
        raise ShadowBaselineBlocked("baseline lineage parent chain is broken")
    if None in (
        entry.proposal_digest,
        entry.approver_id,
        entry.owner_signature_base64,
        entry.owner_approval_digest,
    ):
        raise ShadowBaselineBlocked("baseline lineage advance lacks owner approval evidence")

    if entry.ordinal == 1:
        previous_digest = _EMPTY_DIGEST
        previous_head: str | None = None
    else:
        previous_snapshot = _make_lineage(
            owner_key_fingerprint=lineage.owner_key_fingerprint,
            replay_sha256=lineage.replay_sha256,
            entries=prefix_entries,
        )
        previous_digest = previous_snapshot.lineage_digest
        previous_head = previous.candidate_id

    proposal_payload = _proposal_payload(
        baseline_candidate_id=previous.candidate_id,
        candidate_id=entry.candidate_id,
        replay_sha256=lineage.replay_sha256,
        regression_report_digest=entry.regression_report_digest,
        baseline_scorecard_digest=previous.scorecard_digest,
        baseline_receipt_digest=previous.receipt_digest,
        candidate_scorecard_digest=entry.scorecard_digest,
        candidate_receipt_digest=entry.receipt_digest,
        previous_lineage_digest=previous_digest,
        previous_head_candidate_id=previous_head,
        owner_key_fingerprint=lineage.owner_key_fingerprint,
    )
    expected_proposal = _digest(proposal_payload)
    if entry.proposal_digest != expected_proposal:
        raise ShadowBaselineBlocked("baseline lineage proposal digest is inconsistent")

    signature = entry.owner_signature_base64
    approver = entry.approver_id
    approval_digest = entry.owner_approval_digest
    if not isinstance(signature, str) or not isinstance(approver, str):
        raise ShadowBaselineBlocked("baseline lineage owner approval is malformed")
    approval_payload = _approval_payload(
        candidate_id=entry.candidate_id,
        approver_id=approver,
        owner_key_fingerprint=lineage.owner_key_fingerprint,
        proposal_digest=expected_proposal,
        signature_base64=signature,
    )
    if approval_digest != _digest(approval_payload):
        raise ShadowBaselineBlocked("baseline lineage approval digest is inconsistent")
    _verify_signature(owner_public_key, expected_proposal, approver, signature)


def _advance_entry(
    proposal: ShadowBaselineProposal,
    approval: ShadowBaselineApproval,
) -> ShadowBaselineEntry:
    return ShadowBaselineEntry(
        ordinal=1,
        kind="advance",
        candidate_id=proposal.candidate_id,
        parent_candidate_id=proposal.baseline_candidate_id,
        scorecard_digest=proposal.candidate_scorecard_digest,
        receipt_digest=proposal.candidate_receipt_digest,
        replay_sha256=proposal.replay_sha256,
        regression_report_digest=proposal.regression_report_digest,
        proposal_digest=proposal.proposal_digest,
        approver_id=approval.approver_id,
        owner_signature_base64=approval.signature_base64,
        owner_approval_digest=approval.approval_digest,
    )


def _make_lineage(
    *,
    owner_key_fingerprint: str,
    replay_sha256: str,
    entries: list[ShadowBaselineEntry],
) -> ShadowBaselineLineage:
    if len(entries) < 2:
        raise ShadowBaselineBlocked("baseline lineage needs a seed and an approved advance")
    head = entries[-1]
    payload = {
        "schema_version": "sentinel.shadow-baseline-lineage.v1",
        "state": "owner_curated_shadow_baseline_lineage",
        "authority": "explanation_only",
        "owner_key_fingerprint": owner_key_fingerprint,
        "replay_sha256": replay_sha256,
        "head_candidate_id": head.candidate_id,
        "head_scorecard_digest": head.scorecard_digest,
        "head_receipt_digest": head.receipt_digest,
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "owner_signature_required_for_advance": True,
        "historical_signatures_verified": True,
        "manual_review_required": True,
        "automatic_baseline_selection_allowed": False,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowBaselineLineage.model_validate(
        {**payload, "lineage_digest": _digest(payload)}
    )


def _proposal_payload(
    *,
    baseline_candidate_id: str,
    candidate_id: str,
    replay_sha256: str,
    regression_report_digest: str,
    baseline_scorecard_digest: str,
    baseline_receipt_digest: str,
    candidate_scorecard_digest: str,
    candidate_receipt_digest: str,
    previous_lineage_digest: str,
    previous_head_candidate_id: str | None,
    owner_key_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.shadow-baseline-proposal.v1",
        "state": "awaiting_owner_signature",
        "authority": "explanation_only",
        "baseline_candidate_id": baseline_candidate_id,
        "candidate_id": candidate_id,
        "replay_sha256": replay_sha256,
        "regression_report_digest": regression_report_digest,
        "baseline_scorecard_digest": baseline_scorecard_digest,
        "baseline_receipt_digest": baseline_receipt_digest,
        "candidate_scorecard_digest": candidate_scorecard_digest,
        "candidate_receipt_digest": candidate_receipt_digest,
        "previous_lineage_digest": previous_lineage_digest,
        "previous_head_candidate_id": previous_head_candidate_id,
        "owner_key_fingerprint": owner_key_fingerprint,
        "owner_signature_required": True,
        "manual_review_required": True,
        "automatic_baseline_selection_allowed": False,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }


def _approval_payload(
    *,
    candidate_id: str,
    approver_id: str,
    owner_key_fingerprint: str,
    proposal_digest: str,
    signature_base64: str,
) -> dict[str, object]:
    return {
        "schema_version": "sentinel.shadow-baseline-approval.v1",
        "state": "owner_approved_baseline_advance",
        "authority": "explanation_only",
        "candidate_id": candidate_id,
        "approver_id": approver_id,
        "owner_key_fingerprint": owner_key_fingerprint,
        "proposal_digest": proposal_digest,
        "signature_algorithm": "ed25519",
        "signature_base64": signature_base64,
        "signature_verified": True,
        "automatic_baseline_selection_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }


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


def _verify_signature(
    owner_public_key: Ed25519PublicKey,
    proposal_digest: str,
    approver_id: str,
    signature_base64: str,
) -> None:
    try:
        signature = base64.b64decode(signature_base64, validate=True)
        owner_public_key.verify(
            signature,
            _signature_message(proposal_digest, approver_id),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ShadowBaselineBlocked("baseline owner signature verification failed") from exc


def _signature_message(proposal_digest: str, approver_id: str) -> bytes:
    return (
        _SIGNATURE_CONTEXT
        + proposal_digest.encode("ascii")
        + b"\0"
        + approver_id.encode("utf-8")
    )


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
