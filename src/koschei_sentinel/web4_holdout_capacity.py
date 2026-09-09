from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutRelease,
    verify_web4_holdout_release,
)

_DIGEST = r"^[a-f0-9]{64}$"


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _load_policy(path: str | Path) -> tuple[dict[str, object], str]:
    policy_path = Path(path)
    if policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError("Web4 capacity benchmark policy must be a regular non-symlink file")
    raw = policy_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Web4 capacity benchmark policy JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Web4 capacity benchmark policy must contain one JSON object")
    if payload.get("schema_version") != "sentinel.web4-eval-policy.v1":
        raise ValueError("unsupported Web4 capacity benchmark policy schema")
    if payload.get("status") != "research_only":
        raise ValueError("Web4 capacity benchmark policy must remain research only")
    if payload.get("promotion_eligible") is not False:
        raise ValueError("Web4 capacity benchmark policy must keep promotion disabled")
    if payload.get("training_overlap_allowed") is not False:
        raise ValueError("Web4 capacity benchmark policy must forbid training overlap")
    if payload.get("requires_human_reviewed_holdout") is not True:
        raise ValueError("Web4 capacity benchmark policy must require human-reviewed HOLDOUT")

    desired_total = payload.get("desired_minimum_total_cases")
    minimum_per_family = payload.get("minimum_cases_per_required_family")
    families = payload.get("required_families")
    if not isinstance(desired_total, int) or isinstance(desired_total, bool) or desired_total < 1:
        raise ValueError("Web4 capacity desired minimum total cases must be positive")
    if (
        not isinstance(minimum_per_family, int)
        or isinstance(minimum_per_family, bool)
        or minimum_per_family < 1
    ):
        raise ValueError("Web4 capacity minimum cases per family must be positive")
    if not isinstance(families, list) or not families:
        raise ValueError("Web4 capacity required families are missing")
    if not all(isinstance(item, str) and item for item in families):
        raise ValueError("Web4 capacity required families contain invalid names")
    if len(families) != len(set(families)):
        raise ValueError("Web4 capacity required families contain duplicates")
    return payload, hashlib.sha256(raw).hexdigest()


class Web4HoldoutCapacityReport(StrictModel):
    schema_version: Literal["sentinel.web4-holdout-capacity.v1"] = (
        "sentinel.web4-holdout-capacity.v1"
    )
    release_id: str
    release_sha256: str = Field(pattern=_DIGEST)
    release_artifact_sha256: str = Field(pattern=_DIGEST)
    benchmark_policy_sha256: str = Field(pattern=_DIGEST)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    owner_key_fingerprint: str = Field(pattern=_DIGEST)
    release_verified: Literal[True] = True
    deterministic_holdout_only: Literal[True] = True
    answer_keys_isolated: Literal[True] = True
    all_cases_human_reviewed: Literal[True] = True
    all_cases_independently_adjudicated: Literal[True] = True
    required_family_count: int = Field(gt=0)
    release_case_count: int = Field(gt=0)
    desired_minimum_total_cases: int = Field(gt=0)
    total_case_shortfall: int = Field(ge=0)
    minimum_cases_per_required_family: int = Field(gt=0)
    family_counts: dict[str, int]
    family_shortfalls: dict[str, int]
    covered_family_count: int = Field(ge=0)
    families_at_minimum_count: int = Field(ge=0)
    missing_families: list[str]
    case_ids_sha256: str = Field(pattern=_DIGEST)
    research_benchmark_capacity_ready: bool
    blockers: list[str]
    research_evaluation_only: Literal[True] = True
    model_execution_authorized: Literal[False] = False
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    capacity_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def capacity_contract_verifies(self) -> Web4HoldoutCapacityReport:
        family_names = sorted(self.family_counts)
        if family_names != sorted(self.family_shortfalls):
            raise ValueError("Web4 capacity family count and shortfall keys differ")
        if self.required_family_count != len(family_names):
            raise ValueError("Web4 capacity required_family_count differs from family map")
        if any(value < 0 for value in self.family_counts.values()):
            raise ValueError("Web4 capacity family counts must not be negative")
        if any(value < 0 for value in self.family_shortfalls.values()):
            raise ValueError("Web4 capacity family shortfalls must not be negative")
        if sum(self.family_counts.values()) != self.release_case_count:
            raise ValueError("Web4 capacity family counts differ from release case count")

        expected_total_shortfall = max(
            0,
            self.desired_minimum_total_cases - self.release_case_count,
        )
        if self.total_case_shortfall != expected_total_shortfall:
            raise ValueError("Web4 capacity total case shortfall does not verify")
        expected_shortfalls = {
            family: max(0, self.minimum_cases_per_required_family - count)
            for family, count in self.family_counts.items()
        }
        if self.family_shortfalls != expected_shortfalls:
            raise ValueError("Web4 capacity family shortfalls do not verify")
        expected_missing = sorted(
            family for family, count in self.family_counts.items() if count == 0
        )
        if self.missing_families != expected_missing:
            raise ValueError("Web4 capacity missing family list does not verify")
        expected_covered = sum(count > 0 for count in self.family_counts.values())
        if self.covered_family_count != expected_covered:
            raise ValueError("Web4 capacity covered family count does not verify")
        expected_at_minimum = sum(
            count >= self.minimum_cases_per_required_family
            for count in self.family_counts.values()
        )
        if self.families_at_minimum_count != expected_at_minimum:
            raise ValueError("Web4 capacity minimum-covered family count does not verify")
        expected_ready = (
            self.total_case_shortfall == 0
            and all(shortfall == 0 for shortfall in self.family_shortfalls.values())
            and not self.blockers
        )
        if self.research_benchmark_capacity_ready != expected_ready:
            raise ValueError("Web4 capacity readiness does not match capacity contract")

        payload = self.model_dump(mode="json")
        observed = str(payload.pop("capacity_sha256"))
        if _digest(payload) != observed:
            raise ValueError("Web4 HOLDOUT capacity self-hash does not verify")
        return self


def build_web4_holdout_capacity_report(
    *,
    release: Web4HoldoutRelease,
    owner_public_key: Ed25519PublicKey,
    benchmark_policy_path: str | Path,
) -> Web4HoldoutCapacityReport:
    policy, policy_sha = _load_policy(benchmark_policy_path)
    verified = verify_web4_holdout_release(
        release,
        owner_public_key=owner_public_key,
        benchmark_policy_path=benchmark_policy_path,
    )
    if verified.benchmark_policy_sha256 != policy_sha:
        raise ValueError("Web4 capacity release does not bind supplied benchmark policy")

    required_families = sorted(str(item) for item in policy["required_families"])
    desired_total = int(policy["desired_minimum_total_cases"])
    minimum_per_family = int(policy["minimum_cases_per_required_family"])
    counts = {family: 0 for family in required_families}
    for case in verified.cases:
        if case.family not in counts:
            raise ValueError(f"Web4 capacity release contains unknown family: {case.family}")
        counts[case.family] += 1

    shortfalls = {
        family: max(0, minimum_per_family - count)
        for family, count in counts.items()
    }
    total_shortfall = max(0, desired_total - verified.case_count)
    blockers: list[str] = []
    if total_shortfall:
        blockers.append(f"signed HOLDOUT total case shortfall: {total_shortfall}")
    underfilled = sum(value > 0 for value in shortfalls.values())
    if underfilled:
        blockers.append(f"required family minimum shortfall affects {underfilled} families")

    payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-capacity.v1",
        "release_id": verified.release_id,
        "release_sha256": verified.release_sha256,
        "release_artifact_sha256": verified.artifact_sha256,
        "benchmark_policy_sha256": verified.benchmark_policy_sha256,
        "source_registry_sha256": verified.source_registry_sha256,
        "owner_key_fingerprint": verified.owner_key_fingerprint,
        "release_verified": True,
        "deterministic_holdout_only": verified.deterministic_holdout_only,
        "answer_keys_isolated": verified.answer_keys_isolated,
        "all_cases_human_reviewed": verified.all_cases_human_reviewed,
        "all_cases_independently_adjudicated": verified.all_cases_independently_adjudicated,
        "required_family_count": len(required_families),
        "release_case_count": verified.case_count,
        "desired_minimum_total_cases": desired_total,
        "total_case_shortfall": total_shortfall,
        "minimum_cases_per_required_family": minimum_per_family,
        "family_counts": counts,
        "family_shortfalls": shortfalls,
        "covered_family_count": sum(count > 0 for count in counts.values()),
        "families_at_minimum_count": sum(
            count >= minimum_per_family for count in counts.values()
        ),
        "missing_families": sorted(family for family, count in counts.items() if count == 0),
        "case_ids_sha256": _digest(verified.case_ids),
        "research_benchmark_capacity_ready": not blockers,
        "blockers": blockers,
        "research_evaluation_only": True,
        "model_execution_authorized": False,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
    }
    payload["capacity_sha256"] = _digest(payload)
    return Web4HoldoutCapacityReport.model_validate(payload)
