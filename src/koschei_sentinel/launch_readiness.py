from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.candidate_finalization import (
    CandidateFinalization,
    load_candidate_finalization,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_authority import (
    ProductionAuthority,
    ProductionAuthorityProposal,
    load_verified_production_holdout,
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
        finalization = load_candidate_finalization(finalization_path)
        candidate_id = finalization.candidate_id
        finalized_candidate_valid = True
    except ValueError as exc:
        blockers.append(str(exc))

    valid_holdouts = 0
    holdout_digests: list[str] = []
    holdout_owner_fingerprints: set[str] = set()
    holdout_adapter_digests: set[str] = set()
    if not holdout_evidence_paths:
        blockers.append("no independent holdout evidence was supplied")
    else:
        for path in holdout_evidence_paths:
            try:
                evidence = load_verified_production_holdout(path)
            except ValueError as exc:
                blockers.append(str(exc))
                continue
            valid_holdouts += 1
            holdout_digests.append(evidence.evidence_sha256)
            if evidence.owner_key_fingerprint is not None:
                holdout_owner_fingerprints.add(evidence.owner_key_fingerprint)
            holdout_adapter_digests.add(evidence.adapter_digest)

    if finalization is not None and holdout_adapter_digests:
        if holdout_adapter_digests != {finalization.adapter_digest}:
            blockers.append("Gold HOLDOUT evidence does not bind the finalized adapter")

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
        authority_blockers: list[str] = []
        try:
            proposal = ProductionAuthorityProposal.model_validate(
                _load_json(production_authority_proposal_path, "production authority proposal")
            )
            authority = ProductionAuthority.model_validate(
                _load_json(production_authority_path, "production authority")
            )
            owner_public_key = load_owner_public_key(owner_public_key_path)
            verify_production_authority(proposal, authority, owner_public_key)

            if finalization is None:
                authority_blockers.append(
                    "production authority cannot be bound without valid finalization"
                )
            else:
                if proposal.candidate_id != finalization.candidate_id:
                    authority_blockers.append(
                        "production authority candidate does not match finalization"
                    )
                if proposal.finalization_digest != finalization.finalization_digest:
                    authority_blockers.append(
                        "production authority does not bind candidate finalization"
                    )
            if sorted(proposal.holdout_evidence_digests) != sorted(holdout_digests):
                authority_blockers.append(
                    "production authority does not bind supplied holdout evidence"
                )
            if holdout_owner_fingerprints and holdout_owner_fingerprints != {
                proposal.owner_key_fingerprint
            }:
                authority_blockers.append(
                    "Gold HOLDOUT owner trust root does not match production authority"
                )

            authority_verified = not authority_blockers
            deployment_scope = authority.deployment_scope
            max_initial_traffic_percent = authority.max_initial_traffic_percent
        except (OSError, TypeError, ValueError) as exc:
            authority_blockers.append(str(exc))
        blockers.extend(authority_blockers)

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