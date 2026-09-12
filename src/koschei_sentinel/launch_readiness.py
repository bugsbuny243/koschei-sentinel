from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.candidate_finalization import CandidateFinalization
from koschei_sentinel.models import StrictModel


class LaunchReadinessReport(StrictModel):
    schema_version: Literal["sentinel.launch-readiness.v1"] = "sentinel.launch-readiness.v1"
    candidate_id: str | None = None
    finalized_candidate_valid: bool = False
    holdout_evidence_count: int = Field(default=0, ge=0)
    production_authority_present: bool = False
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
) -> LaunchReadinessReport:
    blockers: list[str] = []
    candidate_id: str | None = None
    finalized_candidate_valid = False

    try:
        finalization = CandidateFinalization.model_validate(
            _load_json(finalization_path, "candidate finalization")
        )
        candidate_id = finalization.candidate_id
        finalized_candidate_valid = True
        if finalization.production_deployment_allowed is not True:
            blockers.append(
                "candidate finalization explicitly forbids production deployment"
            )
        if finalization.automatic_promotion_allowed is not True:
            blockers.append("candidate finalization forbids automatic promotion")
    except ValueError as exc:
        blockers.append(str(exc))

    valid_holdouts = 0
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

    production_authority_present = False
    if production_authority_path is None:
        blockers.append("no explicit production deployment authority artifact was supplied")
    else:
        try:
            authority = _load_json(production_authority_path, "production authority")
            production_authority_present = authority.get("production_deployment_allowed") is True
            if not production_authority_present:
                blockers.append("production authority does not allow deployment")
            if candidate_id is not None:
                authority_candidate = authority.get("candidate_id")
                if authority_candidate != candidate_id:
                    blockers.append("production authority candidate does not match finalization")
        except ValueError as exc:
            blockers.append(str(exc))

    ready = (
        finalized_candidate_valid
        and valid_holdouts > 0
        and production_authority_present
        and not blockers
    )
    return LaunchReadinessReport(
        candidate_id=candidate_id,
        finalized_candidate_valid=finalized_candidate_valid,
        holdout_evidence_count=valid_holdouts,
        production_authority_present=production_authority_present,
        market_release_ready=ready,
        blockers=blockers,
    )
