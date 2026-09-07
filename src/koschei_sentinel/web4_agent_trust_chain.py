from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import atomic_write

_DIGEST = r"^[a-f0-9]{64}$"
_EXPECTED_STAGES = [
    "IDENTITY",
    "DISCOVERY",
    "DELEGATION",
    "EXECUTION_RUNTIME_CONTROL",
    "PROVENANCE",
    "REVOCATION",
]
_REQUIRED_FAMILIES = {
    "dns_agent_discovery_spoofing",
    "endpoint_proof_mismatch",
    "stale_or_revoked_agent_identity",
    "delegation_scope_escalation",
    "confused_deputy_delegation",
    "spawn_chain_provenance_forgery",
    "issuer_key_compromise_valid_signature",
    "provenance_stripping_or_replay",
    "runtime_policy_bypass",
    "memory_context_poisoning_trust_chain",
}


class Web4AgentTrustChainReport(StrictModel):
    schema_version: Literal["sentinel.web4-agent-trust-chain-readiness.v1"] = (
        "sentinel.web4-agent-trust-chain-readiness.v1"
    )
    ready_for_research: bool
    training_authorization: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_activation_allowed: Literal[False] = False
    chain_stage_count: int = Field(ge=1)
    family_count: int = Field(ge=1)
    source_ref_count: int = Field(ge=1)
    source_registry_sha256: str = Field(pattern=_DIGEST)
    protocol_tracking_sha256: str = Field(pattern=_DIGEST)
    benchmark_sha256: str = Field(pattern=_DIGEST)
    trust_chain_map_sha256: str = Field(pattern=_DIGEST)
    violations: list[str]
    report_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def report_contract_verifies(self) -> Web4AgentTrustChainReport:
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("report_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Web4 Agent Trust Chain report self-hash does not verify")
        if self.ready_for_research != (not self.violations):
            raise ValueError("Web4 Agent Trust Chain readiness differs from violations")
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
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} row at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} must be a JSON object")
        rows.append(row)
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
            violations.append("Agent Trust Chain source row lacks source_id")
            continue
        if source_id in by_id:
            violations.append(f"duplicate Agent Trust Chain source_id: {source_id}")
            continue
        by_id[source_id] = row
    return by_id


def _tracking_index(
    tracking: dict[str, object], violations: list[str]
) -> dict[str, dict[str, object]]:
    if tracking.get("schema_version") != "sentinel.web4-protocol-tracking.v1":
        violations.append("Web4 protocol tracking schema mismatch")
    if tracking.get("training_authorization") is not False:
        violations.append("Web4 protocol tracking training authorization is open")
    families = tracking.get("families")
    if not isinstance(families, list):
        violations.append("Web4 protocol tracking families are missing")
        return {}
    by_document: dict[str, dict[str, object]] = {}
    for row in families:
        if not isinstance(row, dict):
            violations.append("Web4 protocol tracking family is invalid")
            continue
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            violations.append("Web4 protocol tracking family lacks document_id")
            continue
        if document_id in by_document:
            violations.append(f"duplicate Web4 tracked document_id: {document_id}")
            continue
        by_document[document_id] = row
    return by_document


def _validate_map(
    trust_map: dict[str, object],
    source_by_id: dict[str, dict[str, object]],
    tracking_by_document: dict[str, dict[str, object]],
    benchmark: dict[str, object],
    violations: list[str],
) -> tuple[int, int, int]:
    if trust_map.get("schema_version") != "sentinel.web4-agent-trust-chain.v1":
        violations.append("Agent Trust Chain schema mismatch")
    if trust_map.get("status") != "research_only":
        violations.append("Agent Trust Chain is not research only")
    for field in (
        "training_authorization",
        "promotion_eligible",
        "production_activation_allowed",
    ):
        if trust_map.get(field) is not False:
            violations.append(f"Agent Trust Chain authority gate is open: {field}")

    stages = trust_map.get("chain_stages")
    if stages != _EXPECTED_STAGES:
        violations.append("Agent Trust Chain stages differ from the six-stage contract")
    stage_set = set(_EXPECTED_STAGES)

    source_refs = trust_map.get("source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        violations.append("Agent Trust Chain source_refs are missing")
        map_sources: set[str] = set()
    else:
        map_sources = {ref for ref in source_refs if isinstance(ref, str) and ref}
        if len(map_sources) != len(source_refs):
            violations.append("Agent Trust Chain source_refs must be unique strings")
    for source_ref in sorted(map_sources):
        source = source_by_id.get(source_ref)
        if source is None:
            violations.append(f"Agent Trust Chain references unknown source: {source_ref}")
            continue
        if source.get("training_authorization") is not False:
            violations.append(f"Agent Trust Chain source training authorization opened: {source_ref}")
        if source.get("eval_exclusion") is not True:
            violations.append(f"Agent Trust Chain source eval exclusion disabled: {source_ref}")
        if source.get("license_status") != "REVIEW_REQUIRED":
            violations.append(f"Agent Trust Chain source bypasses license review: {source_ref}")
        source_type = str(source.get("source_type", ""))
        revision_status = str(source.get("revision_status", ""))
        if "INTERNET_DRAFT" in source_type or "INTERNET_DRAFT" in revision_status:
            if source.get("authority_tier") == "T0_STANDARD":
                violations.append(f"Internet-Draft elevated to standard: {source_ref}")
        if source_type == "W3C_WORKING_DRAFT" and source.get("authority_tier") == "T0_STANDARD":
            violations.append(f"W3C Working Draft elevated to standard: {source_ref}")

    benchmark_families_raw = benchmark.get("required_families")
    if not isinstance(benchmark_families_raw, list):
        violations.append("Web4 benchmark required families are missing")
        benchmark_families: set[str] = set()
    else:
        benchmark_families = {row for row in benchmark_families_raw if isinstance(row, str)}

    families_raw = trust_map.get("families")
    if not isinstance(families_raw, list) or not families_raw:
        violations.append("Agent Trust Chain families are missing")
        return len(_EXPECTED_STAGES), 0, len(map_sources)

    family_ids: set[str] = set()
    for family in families_raw:
        if not isinstance(family, dict):
            violations.append("Agent Trust Chain family must be a JSON object")
            continue
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            violations.append("Agent Trust Chain family lacks family_id")
            continue
        if family_id in family_ids:
            violations.append(f"duplicate Agent Trust Chain family_id: {family_id}")
        family_ids.add(family_id)
        if family.get("stage") not in stage_set:
            violations.append(f"Agent Trust Chain family has unknown stage: {family_id}")
        parent = family.get("benchmark_parent_family")
        if not isinstance(parent, str) or parent not in benchmark_families:
            violations.append(f"Agent Trust Chain family has unknown benchmark parent: {family_id}")
        refs = family.get("source_refs")
        if not isinstance(refs, list) or not refs:
            violations.append(f"Agent Trust Chain family lacks source refs: {family_id}")
        else:
            for ref in refs:
                if not isinstance(ref, str) or ref not in map_sources:
                    violations.append(f"Agent Trust Chain family has unbound source ref: {family_id}")
        behavior = family.get("expected_model_behavior")
        if not isinstance(behavior, str) or not behavior:
            violations.append(f"Agent Trust Chain family lacks defensive behavior: {family_id}")

    missing_required = sorted(_REQUIRED_FAMILIES - family_ids)
    extra_required = sorted(family_ids - _REQUIRED_FAMILIES)
    if missing_required:
        violations.append("Agent Trust Chain required subfamilies are missing")
    if extra_required:
        violations.append("Agent Trust Chain has unreviewed required subfamilies")

    assertions = trust_map.get("revision_assertions")
    if not isinstance(assertions, list) or not assertions:
        violations.append("Agent Trust Chain revision assertions are missing")
    else:
        for assertion in assertions:
            if not isinstance(assertion, dict):
                violations.append("Agent Trust Chain revision assertion is invalid")
                continue
            document_id = assertion.get("document_id")
            revision = assertion.get("verified_observed_revision")
            if not isinstance(document_id, str) or not isinstance(revision, str):
                violations.append("Agent Trust Chain revision assertion lacks identity")
                continue
            tracked = tracking_by_document.get(document_id)
            if tracked is None:
                violations.append(f"Agent Trust Chain assertion references untracked document: {document_id}")
                continue
            if tracked.get("current_observed_revision") != revision:
                violations.append(f"Agent Trust Chain verified revision drifted: {document_id}")

    return len(_EXPECTED_STAGES), len(family_ids), len(map_sources)


def evaluate_web4_agent_trust_chain(
    *,
    source_registry_path: str | Path,
    protocol_tracking_path: str | Path,
    benchmark_path: str | Path,
    trust_chain_map_path: str | Path,
) -> Web4AgentTrustChainReport:
    sources, source_sha = _load_jsonl(Path(source_registry_path), "Web4 source registry")
    tracking, tracking_sha = _load_json(Path(protocol_tracking_path), "Web4 protocol tracking")
    benchmark, benchmark_sha = _load_json(Path(benchmark_path), "Web4 benchmark")
    trust_map, map_sha = _load_json(Path(trust_chain_map_path), "Web4 Agent Trust Chain map")

    violations: list[str] = []
    source_by_id = _source_index(sources, violations)
    tracking_by_document = _tracking_index(tracking, violations)
    stage_count, family_count, source_ref_count = _validate_map(
        trust_map,
        source_by_id,
        tracking_by_document,
        benchmark,
        violations,
    )

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.web4-agent-trust-chain-readiness.v1",
        "ready_for_research": not violations,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
        "chain_stage_count": stage_count,
        "family_count": family_count,
        "source_ref_count": source_ref_count,
        "source_registry_sha256": source_sha,
        "protocol_tracking_sha256": tracking_sha,
        "benchmark_sha256": benchmark_sha,
        "trust_chain_map_sha256": map_sha,
        "violations": violations,
    }
    return Web4AgentTrustChainReport(
        **unsigned,
        report_sha256=_sha256_canonical(unsigned),
    )


def write_web4_agent_trust_chain_report(
    report: Web4AgentTrustChainReport,
    path: str | Path,
) -> None:
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)
