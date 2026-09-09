from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write, canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_PRIORITY_SOURCE_REFS = {
    "ietf.draft.nemethi.aid-agent-identity-discovery-00",
    "ietf.draft.prakash.agent-identity-protocol-01",
    "ietf.draft.tonyai.a2a-trust-01",
    "owasp.agent-control-standard-2026",
}


class Web4AgentTrustAuthoringReport(StrictModel):
    schema_version: Literal["sentinel.web4-agent-trust-authoring-readiness.v1"] = (
        "sentinel.web4-agent-trust-authoring-readiness.v1"
    )
    plan_valid: bool
    training_authorization: Literal[False] = False
    evaluation_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    subfamily_count: int = Field(ge=1)
    p0_subfamily_count: int = Field(ge=1)
    p1_subfamily_count: int = Field(ge=0)
    planning_target_holdout_cases: int = Field(ge=1)
    expected_proposal_count: int = Field(ge=1)
    source_ref_count: int = Field(ge=1)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    trust_chain_map_sha256: str = Field(pattern=_DIGEST)
    intake_policy_sha256: str = Field(pattern=_DIGEST)
    authoring_plan_sha256: str = Field(pattern=_DIGEST)
    violations: list[str]
    report_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def report_contract_verifies(self) -> Web4AgentTrustAuthoringReport:
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("report_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Web4 Agent Trust authoring report self-hash does not verify")
        if self.plan_valid != (not self.violations):
            raise ValueError("Web4 Agent Trust authoring state differs from violations")
        return self


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _load_json(path: Path, label: str) -> tuple[dict[str, object], str]:
    raw = _read_regular(path, label)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_jsonl(path: Path, label: str) -> tuple[list[dict[str, object]], str]:
    raw = _read_regular(path, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} row at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{label} row {line_number} must be a JSON object")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows, hashlib.sha256(raw).hexdigest()


def _source_index(
    rows: list[dict[str, object]], violations: list[str]
) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            violations.append("Agent Trust authoring source lacks source_id")
            continue
        if source_id in by_id:
            violations.append(f"duplicate Agent Trust authoring source_id: {source_id}")
            continue
        by_id[source_id] = row
    return by_id


def _trust_family_index(
    trust_map: dict[str, object], violations: list[str]
) -> dict[str, dict[str, object]]:
    if trust_map.get("schema_version") != "sentinel.web4-agent-trust-chain.v1":
        violations.append("Agent Trust Chain map schema mismatch")
    if trust_map.get("status") != "research_only":
        violations.append("Agent Trust Chain map is not research only")
    for field in ("training_authorization", "promotion_eligible", "production_activation_allowed"):
        if trust_map.get(field) is not False:
            violations.append(f"Agent Trust Chain map authority gate is open: {field}")
    families = trust_map.get("families")
    if not isinstance(families, list) or not families:
        violations.append("Agent Trust Chain map has no families")
        return {}
    by_id: dict[str, dict[str, object]] = {}
    for family in families:
        if not isinstance(family, dict):
            violations.append("Agent Trust Chain family is invalid")
            continue
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            violations.append("Agent Trust Chain family lacks family_id")
            continue
        if family_id in by_id:
            violations.append(f"duplicate Agent Trust Chain family_id: {family_id}")
            continue
        by_id[family_id] = family
    return by_id


def _validate_intake_policy(
    policy: dict[str, object], violations: list[str]
) -> int:
    if policy.get("schema_version") != "sentinel.web4-benchmark-intake-policy.v1":
        violations.append("Web4 benchmark intake policy schema mismatch")
    if policy.get("split_before_human_review") is not True:
        violations.append("Web4 benchmark split must remain before human review")
    if policy.get("manual_split_override_allowed") is not False:
        violations.append("Web4 benchmark manual split override must remain disabled")
    if policy.get("training_authorization") is not False:
        violations.append("Web4 benchmark intake training authorization is open")
    if policy.get("evaluation_authorization_before_human_review") is not False:
        violations.append("Web4 benchmark evaluation authorization is open before review")
    if policy.get("answer_key_storage") != "SEPARATE_FILE_SHA256_ONLY":
        violations.append("Web4 benchmark answer-key isolation contract drifted")
    holdout_bps = policy.get("holdout_bps")
    if not isinstance(holdout_bps, int) or not 1 <= holdout_bps <= 10000:
        violations.append("Web4 benchmark HOLDOUT basis points are invalid")
        return 0
    return holdout_bps


def _validate_plan(
    plan: dict[str, object],
    *,
    source_by_id: dict[str, dict[str, object]],
    trust_by_id: dict[str, dict[str, object]],
    holdout_bps: int,
    violations: list[str],
) -> tuple[int, int, int, int, int, int]:
    if plan.get("schema_version") != "sentinel.web4-agent-trust-authoring-plan.v1":
        violations.append("Agent Trust authoring plan schema mismatch")
    if plan.get("status") != "planning_only":
        violations.append("Agent Trust authoring plan is not planning only")
    for field in (
        "automatic_gold_generation_allowed",
        "automatic_human_review_allowed",
        "training_authorization",
        "evaluation_authorization",
        "promotion_eligible",
        "production_activation_allowed",
    ):
        if plan.get(field) is not False:
            violations.append(f"Agent Trust authoring authority gate is open: {field}")
    for field in (
        "source_license_review_required",
        "source_provenance_review_required",
        "source_snapshot_required",
        "batch_intake_required",
        "split_is_authoritative_only_after_intake",
        "expected_proposal_count_is_not_a_guarantee",
    ):
        if plan.get(field) is not True:
            violations.append(f"Agent Trust authoring safety requirement is disabled: {field}")
    if plan.get("answer_key_storage") != "SEPARATE_FILE_SHA256_ONLY":
        violations.append("Agent Trust authoring answer-key isolation drifted")

    configured_bps = plan.get("planning_holdout_basis_points")
    if configured_bps != holdout_bps:
        violations.append("Agent Trust authoring HOLDOUT rate differs from intake policy")
    target_per_family = plan.get("planning_target_holdout_per_subfamily")
    if not isinstance(target_per_family, int) or target_per_family < 1:
        violations.append("Agent Trust authoring target HOLDOUT per subfamily is invalid")
        target_per_family = 0
    expected_per_family = (
        math.ceil(target_per_family * 10000 / holdout_bps)
        if target_per_family > 0 and holdout_bps > 0
        else 0
    )
    if plan.get("expected_proposals_per_subfamily_for_target_holdout") != expected_per_family:
        violations.append("Agent Trust authoring expected proposal math drifted")

    priority_groups = plan.get("priority_source_groups")
    priority_source_refs: set[str] = set()
    if not isinstance(priority_groups, dict):
        violations.append("Agent Trust authoring priority source groups are missing")
    else:
        p0 = priority_groups.get("P0")
        p1 = priority_groups.get("P1")
        if not isinstance(p0, list) or not isinstance(p1, list):
            violations.append("Agent Trust authoring priority groups must contain P0 and P1 lists")
        else:
            p0_set = {row for row in p0 if isinstance(row, str)}
            p1_set = {row for row in p1 if isinstance(row, str)}
            if p0_set & p1_set:
                violations.append("Agent Trust authoring priority source groups overlap")
            if not _PRIORITY_SOURCE_REFS.issubset(p0_set):
                violations.append("AID/AIP/A2A/ACS sources are not all P0")
            priority_source_refs = p0_set | p1_set

    trust_source_refs_raw = plan.get("subfamilies")
    if not isinstance(trust_source_refs_raw, list) or not trust_source_refs_raw:
        violations.append("Agent Trust authoring subfamilies are missing")
        return 0, 0, 0, 0, 0, len(priority_source_refs)

    seen: set[str] = set()
    p0_count = 0
    p1_count = 0
    expected_total = 0
    target_total = 0
    referenced_sources: set[str] = set()
    for row in trust_source_refs_raw:
        if not isinstance(row, dict):
            violations.append("Agent Trust authoring subfamily is invalid")
            continue
        family_id = row.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            violations.append("Agent Trust authoring subfamily lacks family_id")
            continue
        if family_id in seen:
            violations.append(f"duplicate Agent Trust authoring family_id: {family_id}")
            continue
        seen.add(family_id)
        trust_family = trust_by_id.get(family_id)
        if trust_family is None:
            violations.append(f"Agent Trust authoring family is not in trust map: {family_id}")
            continue
        if row.get("benchmark_parent_family") != trust_family.get("benchmark_parent_family"):
            violations.append(f"Agent Trust authoring benchmark parent drifted: {family_id}")
        plan_refs = row.get("required_source_refs")
        trust_refs = trust_family.get("source_refs")
        if not isinstance(plan_refs, list) or set(plan_refs) != set(trust_refs or []):
            violations.append(f"Agent Trust authoring source refs drifted: {family_id}")
            plan_ref_set: set[str] = set()
        else:
            plan_ref_set = {ref for ref in plan_refs if isinstance(ref, str)}
        referenced_sources.update(plan_ref_set)
        for source_ref in plan_ref_set:
            source = source_by_id.get(source_ref)
            if source is None:
                violations.append(f"Agent Trust authoring references unknown source: {source_ref}")
                continue
            if source.get("training_authorization") is not False:
                violations.append(f"Agent Trust authoring source training gate is open: {source_ref}")
            if source.get("eval_exclusion") is not True:
                violations.append(f"Agent Trust authoring source eval exclusion is disabled: {source_ref}")
            if source.get("license_status") != "REVIEW_REQUIRED":
                violations.append(f"Agent Trust authoring source bypasses license review: {source_ref}")
        minimum_sources = row.get("minimum_distinct_sources_per_case")
        if not isinstance(minimum_sources, int) or not 1 <= minimum_sources <= len(plan_ref_set):
            violations.append(f"Agent Trust authoring source diversity target is invalid: {family_id}")
        priority = row.get("priority")
        if priority == "P0":
            p0_count += 1
        elif priority == "P1":
            p1_count += 1
        else:
            violations.append(f"Agent Trust authoring priority is invalid: {family_id}")
        target = row.get("planning_target_holdout")
        expected = row.get("expected_proposals_for_target_holdout")
        if target != target_per_family:
            violations.append(f"Agent Trust authoring HOLDOUT target drifted: {family_id}")
        if expected != expected_per_family:
            violations.append(f"Agent Trust authoring proposal expectation drifted: {family_id}")
        if isinstance(target, int):
            target_total += target
        if isinstance(expected, int):
            expected_total += expected

    if seen != set(trust_by_id):
        violations.append("Agent Trust authoring subfamilies differ from trust-chain map")
    if priority_source_refs != referenced_sources:
        violations.append("Agent Trust authoring priority source inventory differs from subfamily sources")

    return (
        len(seen),
        p0_count,
        p1_count,
        target_total,
        expected_total,
        len(referenced_sources),
    )


def evaluate_web4_agent_trust_authoring_plan(
    *,
    source_registry_path: str | Path,
    trust_chain_map_path: str | Path,
    intake_policy_path: str | Path,
    authoring_plan_path: str | Path,
) -> Web4AgentTrustAuthoringReport:
    sources, source_sha = _load_jsonl(Path(source_registry_path), "Web4 source registry")
    trust_map, trust_sha = _load_json(Path(trust_chain_map_path), "Web4 Agent Trust Chain map")
    intake_policy, intake_sha = _load_json(Path(intake_policy_path), "Web4 benchmark intake policy")
    plan, plan_sha = _load_json(Path(authoring_plan_path), "Web4 Agent Trust authoring plan")

    violations: list[str] = []
    source_by_id = _source_index(sources, violations)
    trust_by_id = _trust_family_index(trust_map, violations)
    holdout_bps = _validate_intake_policy(intake_policy, violations)
    subfamilies, p0_count, p1_count, target_total, expected_total, source_count = _validate_plan(
        plan,
        source_by_id=source_by_id,
        trust_by_id=trust_by_id,
        holdout_bps=holdout_bps,
        violations=violations,
    )

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.web4-agent-trust-authoring-readiness.v1",
        "plan_valid": not violations,
        "training_authorization": False,
        "evaluation_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
        "subfamily_count": subfamilies,
        "p0_subfamily_count": p0_count,
        "p1_subfamily_count": p1_count,
        "planning_target_holdout_cases": target_total,
        "expected_proposal_count": expected_total,
        "source_ref_count": source_count,
        "source_registry_sha256": source_sha,
        "trust_chain_map_sha256": trust_sha,
        "intake_policy_sha256": intake_sha,
        "authoring_plan_sha256": plan_sha,
        "violations": violations,
    }
    return Web4AgentTrustAuthoringReport(
        **unsigned,
        report_sha256=_sha256_canonical(unsigned),
    )


def write_web4_agent_trust_authoring_report(
    report: Web4AgentTrustAuthoringReport,
    path: str | Path,
) -> None:
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)
