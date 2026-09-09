from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write

_DIGEST = r"^[a-f0-9]{64}$"


class Web4ResearchReadinessReport(StrictModel):
    schema_version: Literal["sentinel.web4-research-readiness.v1"] = (
        "sentinel.web4-research-readiness.v1"
    )
    ready_for_research: bool
    training_allowed: Literal[False] = False
    source_count: int = Field(ge=1)
    protocol_family_count: int = Field(ge=1)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    protocol_tracking_sha256: str = Field(pattern=_DIGEST)
    curriculum_sha256: str = Field(pattern=_DIGEST)
    benchmark_sha256: str = Field(pattern=_DIGEST)
    event_schema_sha256: str = Field(pattern=_DIGEST)
    violations: list[str]
    report_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def report_contract_verifies(self) -> Web4ResearchReadinessReport:
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("report_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Web4 research readiness self-hash does not verify")
        if self.ready_for_research != (not self.violations):
            raise ValueError("Web4 research readiness state differs from violations")
        return self


def _sha256_canonical(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _validate_sources(
    rows: list[dict[str, object]],
    violations: list[str],
) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            violations.append("Web4 source row lacks source_id")
            continue
        if source_id in by_id:
            violations.append(f"duplicate Web4 source_id: {source_id}")
            continue
        by_id[source_id] = row

        if row.get("schema_version") != "sentinel.web4-source.v1":
            violations.append(f"Web4 source schema mismatch: {source_id}")
        if row.get("training_authorization") is not False:
            violations.append(f"Web4 source training authorization is not closed: {source_id}")
        if row.get("eval_exclusion") is not True:
            violations.append(f"Web4 source eval exclusion is not enabled: {source_id}")

        source_type = str(row.get("source_type", ""))
        revision_status = str(row.get("revision_status", ""))
        is_draft = "INTERNET_DRAFT" in source_type or "INTERNET_DRAFT" in revision_status
        if is_draft and row.get("authority_tier") == "T0_STANDARD":
            violations.append(f"Internet-Draft misclassified as T0 standard: {source_id}")
        if is_draft and source_type == "WEB_STANDARD":
            violations.append(f"Internet-Draft misclassified as web standard: {source_id}")
    return by_id


def _validate_tracking(
    tracking: dict[str, object],
    source_by_id: dict[str, dict[str, object]],
    violations: list[str],
) -> int:
    if tracking.get("schema_version") != "sentinel.web4-protocol-tracking.v1":
        violations.append("Web4 protocol tracking schema mismatch")
    if tracking.get("training_authorization") is not False:
        violations.append("Web4 protocol tracking training authorization is not closed")

    families = tracking.get("families")
    if not isinstance(families, list) or not families:
        violations.append("Web4 protocol tracking has no protocol families")
        return 0

    family_ids: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            violations.append("Web4 protocol family is not a JSON object")
            continue
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            violations.append("Web4 protocol family lacks family_id")
            continue
        if family_id in family_ids:
            violations.append(f"duplicate Web4 protocol family_id: {family_id}")
        family_ids.add(family_id)

        source_ref = family.get("source_ref")
        if not isinstance(source_ref, str) or source_ref not in source_by_id:
            violations.append(f"Web4 protocol family has unknown source_ref: {family_id}")
            continue
        source = source_by_id[source_ref]
        standing = str(family.get("standing", ""))
        if "INTERNET_DRAFT" in standing:
            revision_status = str(source.get("revision_status", ""))
            if "INTERNET_DRAFT" not in revision_status:
                violations.append(f"Web4 draft family source status drifted: {family_id}")

    collisions = tracking.get("known_name_collisions", [])
    if not isinstance(collisions, list):
        violations.append("Web4 known_name_collisions must be a list")
        return len(family_ids)
    for collision in collisions:
        if not isinstance(collision, dict):
            violations.append("Web4 protocol name collision row is invalid")
            continue
        collision_ids = collision.get("family_ids")
        if not isinstance(collision_ids, list) or len(collision_ids) < 2:
            violations.append("Web4 protocol name collision must bind at least two families")
            continue
        unknown = sorted(
            item for item in collision_ids if not isinstance(item, str) or item not in family_ids
        )
        if unknown:
            violations.append("Web4 protocol name collision references unknown families")
    return len(family_ids)


def _validate_curriculum_and_benchmark(
    curriculum: dict[str, object],
    benchmark: dict[str, object],
    violations: list[str],
) -> None:
    if curriculum.get("schema") != "sentinel.curriculum.web4.v1":
        violations.append("Web4 curriculum schema mismatch")
    if curriculum.get("status") != "offline_research_only":
        violations.append("Web4 curriculum is not offline research only")
    if curriculum.get("runtime_integration") is not False:
        violations.append("Web4 curriculum runtime integration must remain disabled")
    if curriculum.get("training_run_allowed") is not False:
        violations.append("Web4 curriculum training gate must remain closed")
    if curriculum.get("training_data_requires_provenance_review") is not True:
        violations.append("Web4 curriculum provenance review gate is disabled")
    if curriculum.get("training_data_requires_license_review") is not True:
        violations.append("Web4 curriculum license review gate is disabled")

    if benchmark.get("schema_version") != "sentinel.web4-eval-policy.v1":
        violations.append("Web4 benchmark schema mismatch")
    if benchmark.get("status") != "research_only":
        violations.append("Web4 benchmark is not research only")
    if benchmark.get("promotion_eligible") is not False:
        violations.append("Web4 benchmark must not be promotion eligible yet")
    if benchmark.get("training_overlap_allowed") is not False:
        violations.append("Web4 benchmark training overlap must remain forbidden")
    if benchmark.get("requires_human_reviewed_holdout") is not True:
        violations.append("Web4 benchmark must require a human-reviewed HOLDOUT")

    curriculum_families = curriculum.get("required_eval_families")
    benchmark_families = benchmark.get("required_families")
    if not isinstance(curriculum_families, list) or not isinstance(benchmark_families, list):
        violations.append("Web4 eval family lists are missing")
    elif set(curriculum_families) != set(benchmark_families):
        violations.append("Web4 curriculum and benchmark eval families differ")

    hard_gates = benchmark.get("hard_gates")
    if not isinstance(hard_gates, dict):
        violations.append("Web4 benchmark hard gates are missing")
        return
    for gate in (
        "authority_boundary_violation_rate",
        "invented_authority_rate",
        "unsupported_standard_claim_rate",
        "answer_key_leakage_rate",
    ):
        if hard_gates.get(gate) != 0.0:
            violations.append(f"Web4 zero-tolerance gate is not zero: {gate}")


def _validate_event_schema(event_schema: dict[str, object], violations: list[str]) -> None:
    properties = event_schema.get("properties")
    if not isinstance(properties, dict):
        violations.append("Web4 event schema properties are missing")
        return
    schema_version = properties.get("schema_version")
    if not isinstance(schema_version, dict):
        violations.append("Web4 event schema version property is invalid")
    elif schema_version.get("const") != "sentinel.web4-security-event.v1":
        violations.append("Web4 event schema version const mismatch")
    if event_schema.get("additionalProperties") is not False:
        violations.append("Web4 event schema must reject unknown top-level fields")

    required = event_schema.get("required")
    required_fields = {"entities", "intent", "capabilities", "actions", "consequences", "evidence"}
    if not isinstance(required, list) or not required_fields.issubset(set(required)):
        violations.append("Web4 event schema lacks required security representation fields")


def evaluate_web4_research_readiness(
    *,
    source_registry_path: str | Path,
    protocol_tracking_path: str | Path,
    curriculum_path: str | Path,
    benchmark_path: str | Path,
    event_schema_path: str | Path,
) -> Web4ResearchReadinessReport:
    sources, source_sha = _load_jsonl(Path(source_registry_path), "Web4 source registry")
    tracking, tracking_sha = _load_json(Path(protocol_tracking_path), "Web4 protocol tracking")
    curriculum, curriculum_sha = _load_json(Path(curriculum_path), "Web4 curriculum")
    benchmark, benchmark_sha = _load_json(Path(benchmark_path), "Web4 benchmark")
    event_schema, event_schema_sha = _load_json(Path(event_schema_path), "Web4 event schema")

    violations: list[str] = []
    source_by_id = _validate_sources(sources, violations)
    protocol_family_count = _validate_tracking(tracking, source_by_id, violations)
    _validate_curriculum_and_benchmark(curriculum, benchmark, violations)
    _validate_event_schema(event_schema, violations)

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.web4-research-readiness.v1",
        "ready_for_research": not violations,
        "training_allowed": False,
        "source_count": len(source_by_id),
        "protocol_family_count": protocol_family_count,
        "source_registry_sha256": source_sha,
        "protocol_tracking_sha256": tracking_sha,
        "curriculum_sha256": curriculum_sha,
        "benchmark_sha256": benchmark_sha,
        "event_schema_sha256": event_schema_sha,
        "violations": violations,
    }
    return Web4ResearchReadinessReport(
        **unsigned,
        report_sha256=_sha256_canonical(unsigned),
    )


def write_web4_research_readiness_report(
    report: Web4ResearchReadinessReport,
    path: str | Path,
) -> None:
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)
