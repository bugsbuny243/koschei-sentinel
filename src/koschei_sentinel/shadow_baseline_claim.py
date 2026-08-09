from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.shadow_baseline import (
    ShadowBaselineBlocked,
    ShadowBaselineLineage,
    ShadowBaselineProposal,
)

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class ShadowBaselineSuccessorClaim(StrictModel):
    schema_version: Literal["sentinel.shadow-baseline-successor-claim.v1"] = (
        "sentinel.shadow-baseline-successor-claim.v1"
    )
    state: Literal["successor_consumed"] = "successor_consumed"
    authority: Literal["explanation_only"] = "explanation_only"
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    replay_sha256: str = Field(pattern=_DIGEST)
    previous_lineage_digest: str = Field(pattern=_DIGEST)
    previous_head_candidate_id: str | None = Field(default=None, pattern=_CANDIDATE_ID)
    successor_candidate_id: str = Field(pattern=_CANDIDATE_ID)
    successor_lineage_digest: str = Field(pattern=_DIGEST)
    proposal_digest: str = Field(pattern=_DIGEST)
    automatic_baseline_selection_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    claim_digest: str = Field(pattern=_DIGEST)


def build_shadow_baseline_successor_claim(
    proposal: ShadowBaselineProposal,
    successor: ShadowBaselineLineage,
) -> ShadowBaselineSuccessorClaim:
    if successor.owner_key_fingerprint != proposal.owner_key_fingerprint:
        raise ShadowBaselineBlocked("successor lineage owner does not match proposal")
    if successor.replay_sha256 != proposal.replay_sha256:
        raise ShadowBaselineBlocked("successor lineage replay does not match proposal")
    if successor.head_candidate_id != proposal.candidate_id:
        raise ShadowBaselineBlocked("successor lineage head does not match proposal candidate")
    if successor.head_scorecard_digest != proposal.candidate_scorecard_digest:
        raise ShadowBaselineBlocked("successor lineage scorecard does not match proposal")
    if successor.head_receipt_digest != proposal.candidate_receipt_digest:
        raise ShadowBaselineBlocked("successor lineage receipt does not match proposal")

    payload = {
        "schema_version": "sentinel.shadow-baseline-successor-claim.v1",
        "state": "successor_consumed",
        "authority": "explanation_only",
        "owner_key_fingerprint": proposal.owner_key_fingerprint,
        "replay_sha256": proposal.replay_sha256,
        "previous_lineage_digest": proposal.previous_lineage_digest,
        "previous_head_candidate_id": proposal.previous_head_candidate_id,
        "successor_candidate_id": proposal.candidate_id,
        "successor_lineage_digest": successor.lineage_digest,
        "proposal_digest": proposal.proposal_digest,
        "automatic_baseline_selection_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return ShadowBaselineSuccessorClaim.model_validate(
        {**payload, "claim_digest": _digest(payload)}
    )


def claim_shadow_baseline_successor(
    claim: ShadowBaselineSuccessorClaim,
    claim_dir: str | Path,
) -> Path:
    _require_claim_digest(claim)
    root = Path(claim_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / _claim_filename(claim)
    serialized = json.dumps(claim.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    if destination.exists():
        _accept_same_or_reject_fork(destination, claim)
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
            _accept_same_or_reject_fork(destination, claim)
        return destination
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_shadow_baseline_successor_claim(path: str | Path) -> ShadowBaselineSuccessorClaim:
    try:
        claim = ShadowBaselineSuccessorClaim.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid baseline successor claim") from exc
    _require_claim_digest(claim)
    return claim


def _accept_same_or_reject_fork(
    destination: Path,
    expected: ShadowBaselineSuccessorClaim,
) -> None:
    existing = load_shadow_baseline_successor_claim(destination)
    if existing.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ShadowBaselineBlocked(
            "baseline predecessor already has a different claimed successor"
        )


def _claim_filename(claim: ShadowBaselineSuccessorClaim) -> str:
    return (
        f"{claim.owner_key_fingerprint}."
        f"{claim.replay_sha256}."
        f"{claim.previous_lineage_digest}.json"
    )


def _require_claim_digest(claim: ShadowBaselineSuccessorClaim) -> None:
    payload = claim.model_dump(mode="json")
    claimed = payload.pop("claim_digest")
    if claimed != _digest(payload):
        raise ShadowBaselineBlocked("baseline successor claim digest does not match contents")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ShadowBaselineSuccessorClaim",
    "build_shadow_baseline_successor_claim",
    "claim_shadow_baseline_successor",
    "load_shadow_baseline_successor_claim",
]
