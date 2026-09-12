from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.candidate_finalization import CandidateFinalization
from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_authority import (
    ProductionAuthority,
    ProductionAuthorityProposal,
    canonical_json_digest,
    verify_production_authority,
)
from koschei_sentinel.promotion import load_owner_public_key


class LaunchReadinessReport(StrictModel):
    schema_version: Literal["sentinel.launch-readiness.v1"] = "sentinel.launch-readiness.v1"
    candidate_id: str | None = None
    finalized_candidate_valid: bool = False
    holdout_evidence_count: int = Field(default=0, ge=0)
    production_authority_present: bool = False
    production_authority_verified: bool = False
    deployment_scope: str | None = None
    max_initial_traffic_percent: int | None = None
    market_release_ready: bool = False
    blockers: list[str] = Field(default_factory=list)


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{label} is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {source}")
    return payload


def audit_launch_readiness(
    *,
    finalization_path: str | Path,
    holdout_evidence_paths: list[str | Path],
    production_authority_path: str | Path | None = None,
    production_authority_proposal_path: str | Path | None = None,
    owner_public_key_path: str | Path | None = None,
) -> LaunchReadinessReport:
    blockers: list[str] = []
    candidate_id: str | None = None
    finalized_candidate_valid = False
    finalization: CandidateFinalization | None = None

    try:
        finalization = CandidateFinalization.model_validate(
            _load_json(finalization_path, "candidate finalization")
        )
        candidate_id = finalization.candidate_id
        finalized_candidate_valid = True
    except ValueError as exc:
        blockers.append(str(exc))

    valid_holdouts = 0
    holdout_digests: list[str] = []
    if not holdout_evidence_paths:
        blockers.append("no independent holdout evidence was supplied")
    else:
        for path in holdout_evidence_paths:
            try:
                payload = _load_json(path, "holdout evidence")
            except ValueError as exc:
                blockers.append(str(exc))
                continue
            schema = payload.get("schema_version")
            if not isinstance(schema, str) or "holdout" not in schema.lower():
                blockers.append(f"unrecognized holdout evidence schema: {path}")
                continue
            valid_holdouts += 1
            holdout_digests.append(canonical_json_digest(payload))

    authority_present = production_authority_path is not None
    authority_verified = False
    deployment_scope: str | None = None
    max_initial_traffic_percent: int | None = None

    if production_authority_path is None:
        blockers.append("no explicit signed production deployment authority artifact was supplied")
    elif production_authority_proposal_path is None:
        blockers.append("production authority proposal is required for signature verification")
    elif owner_public_key_path is None:
        blockers.append("owner public key is required for production authority verification")
    else:
        try:
            proposal = ProductionAuthorityProposal.model_validate(
                _load_json(production_authority_proposal_path, "production authority proposal")
            )
            authority = ProductionAuthority.model_validate(
                _load_json(production_authority_path, "production authority")
            )
            verify_production_authority(
                proposal,
                authority,
                load_owner_public_key(owner_public_key_path),
            )
            if finalization is None:
                blockers.append("production authority cannot be bound without valid finalization")
            else:
                if proposal.candidate_id != finalization.candidate_id:
                    blockers.append("production authority candidate does not match finalization")
                if proposal.finalization_digest != finalization.finalization_digest:
                    blockers.append("production authority does not bind candidate finalization")
            if sorted(proposal.holdout_evidence_digests) != sorted(holdout_digests):
                blockers.append("production authority does not bind supplied holdout evidence")
            if not authority.rollback_required:
                blockers.append("production authority requires rollback capability")
            if not authority.emergency_disable_required:
                blockers.append("production authority requires emergency disable capability")
            if authority.automatic_expansion_allowed:
                blockers.append("automatic production traffic expansion is forbidden")
            authority_verified = not any(
                "production authority" in item or "rollback" in item or "emergency" in item
                for item in blockers
            )
            deployment_scope = authority.deployment_scope
            max_initial_traffic_percent = authority.max_initial_traffic_percent
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(str(exc))

    ready = (
        finalized_candidate_valid
        and valid_holdouts > 0
        and authority_present
        and authority_verified
        and not blockers
    )
    return LaunchReadinessReport(
        candidate_id=candidate_id,
        finalized_candidate_valid=finalized_candidate_valid,
        holdout_evidence_count=valid_holdouts,
        production_authority_present=authority_present,
        production_authority_verified=authority_verified,
        deployment_scope=deployment_scope,
        max_initial_traffic_percent=max_initial_traffic_percent,
        market_release_ready=ready,
        blockers=blockers,
    )
