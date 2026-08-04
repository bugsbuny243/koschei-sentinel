from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from koschei_sentinel.dataset_build import (
    DatasetBuildManifest,
    build_dataset_release_from_records,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.readiness import ReadinessPolicy
from koschei_sentinel.split import SplitConfig

_DATABASE_URL_ENV = "NEON_DATABASE_URL"
_RULE_ID = re.compile(r"^[A-Z][A-Z0-9._:-]{0,127}$")
_KIND_CHARS = re.compile(r"[^a-z0-9._-]+")
_ALLOWED_GRADES = {"A", "B", "C", "D", "F", "-"}

_REQUIRED_COLUMNS = {
    "security_unified_radar_verdicts": {
        "id",
        "network",
        "target_kind",
        "target_id",
        "grade",
        "verdict",
        "signed",
        "signature",
        "fingerprint",
        "triggered_rules",
        "updated_at",
    },
    "security_radar_verdicts": {
        "id",
        "network",
        "target",
        "module_id",
        "signature",
        "risk_index",
        "risk_level",
        "grade",
        "verdict",
        "recommendation",
        "evidence",
        "signals",
        "rule_version",
        "signed",
        "updated_at",
    },
    "security_radar_holder_snapshots": {
        "id",
        "network",
        "target",
        "owner_wallet",
        "holder_rank",
        "percentage",
        "scanned_at",
    },
}

_SCHEMA_SQL = """
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN (
    'security_unified_radar_verdicts',
    'security_radar_verdicts',
    'security_radar_holder_snapshots'
  )
ORDER BY table_name, ordinal_position
"""

_SOURCE_SQL = """
WITH unified_latest AS (
  SELECT DISTINCT ON (network, target_id)
    id,
    network,
    target_kind,
    target_id,
    grade,
    verdict,
    signature,
    fingerprint,
    triggered_rules,
    updated_at
  FROM public.security_unified_radar_verdicts
  WHERE signed IS TRUE
    AND target_id IS NOT NULL
    AND target_id <> ''
  ORDER BY network, target_id, updated_at DESC, id DESC
), selected AS (
  SELECT *
  FROM unified_latest
  ORDER BY updated_at DESC, network, target_id
  LIMIT %s
), module_latest AS (
  SELECT DISTINCT ON (r.network, r.target, r.module_id)
    r.id,
    r.network,
    r.target,
    r.module_id,
    r.signature,
    r.risk_index,
    r.risk_level,
    r.grade,
    r.verdict,
    r.recommendation,
    r.evidence,
    r.signals,
    r.rule_version,
    r.updated_at
  FROM public.security_radar_verdicts r
  JOIN selected s
    ON s.network = r.network
   AND s.target_id = r.target
  WHERE r.signed IS TRUE
  ORDER BY r.network, r.target, r.module_id, r.updated_at DESC, r.id DESC
), module_agg AS (
  SELECT
    network,
    target,
    jsonb_agg(
      jsonb_build_object(
        'id', id,
        'module_id', module_id,
        'signature', signature,
        'risk_index', risk_index,
        'risk_level', risk_level,
        'grade', grade,
        'verdict', verdict,
        'recommendation', recommendation,
        'evidence', evidence,
        'signals', signals,
        'rule_version', rule_version,
        'updated_at', updated_at
      )
      ORDER BY module_id
    ) AS modules
  FROM module_latest
  GROUP BY network, target
), holder_latest AS (
  SELECT DISTINCT ON (h.network, h.target, h.owner_wallet)
    h.network,
    h.target,
    h.owner_wallet,
    h.holder_rank,
    h.percentage,
    h.scanned_at,
    h.id
  FROM public.security_radar_holder_snapshots h
  JOIN selected s
    ON s.network = h.network
   AND s.target_id = h.target
  WHERE h.holder_rank <= 10
  ORDER BY h.network, h.target, h.owner_wallet, h.scanned_at DESC, h.id DESC
), holder_agg AS (
  SELECT
    network,
    target,
    count(*) AS holder_count,
    max(percentage) AS top_percentage,
    max(scanned_at) AS scanned_at
  FROM holder_latest
  GROUP BY network, target
)
SELECT
  s.id,
  s.network,
  s.target_kind,
  s.target_id,
  s.grade,
  s.verdict,
  s.signature,
  s.fingerprint,
  s.triggered_rules,
  s.updated_at,
  COALESCE(m.modules, '[]'::jsonb) AS modules,
  COALESCE(h.holder_count, 0) AS holder_count,
  h.top_percentage,
  h.scanned_at AS holder_scanned_at
FROM selected s
LEFT JOIN module_agg m
  ON m.network = s.network
 AND m.target = s.target_id
LEFT JOIN holder_agg h
  ON h.network = s.network
 AND h.target = s.target_id
ORDER BY s.network, s.target_id
"""


class NeonSourceStats(StrictModel):
    schema_version: Literal["sentinel.neon-source-stats.v1"] = (
        "sentinel.neon-source-stats.v1"
    )
    selected_targets: int
    emitted_records: int
    rule_evidence_items: int
    module_evidence_items: int
    holder_evidence_items: int
    fallback_evidence_items: int
    source_tables: list[str] = Field(min_length=3, max_length=3)


class NeonDatasetBuildResult(StrictModel):
    schema_version: Literal["sentinel.neon-dataset-build.v1"] = (
        "sentinel.neon-dataset-build.v1"
    )
    source: NeonSourceStats
    build: DatasetBuildManifest


def build_neon_dataset_release(
    *,
    output_dir: str | Path | None,
    salt_version: str,
    limit: int = 5000,
    policy: ReadinessPolicy | None = None,
    split_config: SplitConfig | None = None,
    dry_run: bool = False,
) -> NeonDatasetBuildResult:
    database_url = os.getenv(_DATABASE_URL_ENV, "")
    if not database_url:
        raise ValueError(f"{_DATABASE_URL_ENV} is required")
    records, stats = read_neon_records(database_url, limit=limit)
    build = build_dataset_release_from_records(
        records,
        input_files=1,
        output_dir=output_dir,
        salt_version=salt_version,
        policy=policy,
        split_config=split_config,
        dry_run=dry_run,
    )
    return NeonDatasetBuildResult(source=stats, build=build)


def read_neon_records(
    database_url: str,
    *,
    limit: int = 5000,
) -> tuple[list[dict[str, Any]], NeonSourceStats]:
    if not database_url:
        raise ValueError("database URL is empty")
    if limit < 1 or limit > 10_000:
        raise ValueError("limit must be between 1 and 10000")

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "Neon dependencies are missing; install the project with .[neon]"
        ) from exc

    try:
        with psycopg.connect(
            database_url,
            connect_timeout=15,
            application_name="koschei-sentinel-neon-readonly",
            row_factory=dict_row,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute("SET LOCAL statement_timeout = '120s'")
                cursor.execute(_SCHEMA_SQL)
                _validate_schema(cursor.fetchall())
                cursor.execute(_SOURCE_SQL, (limit,))
                rows = cursor.fetchall()
    except ValueError:
        raise
    except Exception:
        raise RuntimeError(
            "Neon dataset read failed; provider details were intentionally suppressed"
        ) from None

    records, stats = records_from_neon_rows(rows)
    if not records:
        raise ValueError("Neon source returned no signed unified verdicts")
    return records, stats


def records_from_neon_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], NeonSourceStats]:
    records: list[dict[str, Any]] = []
    rule_items = 0
    module_items = 0
    holder_items = 0
    fallback_items = 0

    for row in rows:
        record, counts = _record_from_row(row)
        records.append(record)
        rule_items += counts["rule"]
        module_items += counts["module"]
        holder_items += counts["holder"]
        fallback_items += counts["fallback"]

    stats = NeonSourceStats(
        selected_targets=len(records),
        emitted_records=len(records),
        rule_evidence_items=rule_items,
        module_evidence_items=module_items,
        holder_evidence_items=holder_items,
        fallback_evidence_items=fallback_items,
        source_tables=sorted(_REQUIRED_COLUMNS),
    )
    return records, stats


def _record_from_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    network = _required_text(row.get("network"), "network")
    target = _required_text(row.get("target_id"), "target_id")
    source_id = _required_text(row.get("id"), "id")
    signature = _first_text(
        [row.get("signature"), row.get("fingerprint"), source_id]
    )
    grade = str(row.get("grade") or "-").strip().upper()
    if grade not in _ALLOWED_GRADES:
        grade = "-"

    evidence: list[dict[str, Any]] = []
    triggered_rule_ids: list[str] = []
    counts = {"rule": 0, "module": 0, "holder": 0, "fallback": 0}

    for index, rule in enumerate(_as_list(row.get("triggered_rules"))):
        if not isinstance(rule, Mapping):
            continue
        rule_id = _normalized_rule_id(rule.get("rule_id"))
        if rule_id is None:
            continue
        triggered_rule_ids.append(rule_id)
        status = str(rule.get("evidence_status") or "").strip().lower()
        evidence_keys = _as_list(rule.get("evidence_keys"))
        evidence.append(
            {
                "evidence_id": f"neon-rule:{source_id}:{rule_id}:{index}",
                "kind": "unified_rule",
                "statement": _first_text(
                    [
                        rule.get("summary"),
                        rule.get("title"),
                        f"Unified radar rule {rule_id} was triggered.",
                    ]
                ),
                "confidence": _confidence_from_status(status),
                "rule_ids": [rule_id],
                "attributes": {
                    "source": "security_unified_radar_verdicts",
                    "evidence_status": status or "unknown",
                    "evidence_key_count": len(evidence_keys),
                    "rule_title": _optional_text(rule.get("title")),
                    "tier": _optional_text(rule.get("tier")),
                    "grade_effect": _optional_text(rule.get("grade_effect")),
                },
            }
        )
        counts["rule"] += 1

    for module in _as_list(row.get("modules")):
        if not isinstance(module, Mapping):
            continue
        module_id = _kind(module.get("module_id"), fallback="radar_module")
        module_signature = _first_text(
            [module.get("signature"), module.get("id"), source_id]
        )
        signal_status = ""
        signals = _as_mapping(module.get("signals"))
        if signals:
            signal_status = str(signals.get("evidence_status") or "").strip().lower()
        module_evidence = _as_list(module.get("evidence"))
        evidence.append(
            {
                "evidence_id": f"neon-module:{module_signature}:{module_id}",
                "kind": module_id,
                "statement": _first_text(
                    [
                        *_text_items(module_evidence),
                        module.get("verdict"),
                        module.get("recommendation"),
                        f"Signed radar module {module_id} produced a bounded result.",
                    ]
                ),
                "confidence": _confidence_from_status(signal_status),
                "rule_ids": [],
                "attributes": {
                    "source": "security_radar_verdicts",
                    "module_id": module_id,
                    "risk_index": _primitive_number(module.get("risk_index")),
                    "risk_level": _optional_text(module.get("risk_level")),
                    "module_grade": _optional_text(module.get("grade")),
                    "rule_version": _optional_text(module.get("rule_version")),
                    "evidence_count": len(module_evidence),
                },
            }
        )
        counts["module"] += 1

    holder_count = _primitive_int(row.get("holder_count")) or 0
    top_percentage = _primitive_number(row.get("top_percentage"))
    if holder_count > 0:
        statement = f"The latest bounded holder snapshot resolved {holder_count} holders."
        if top_percentage is not None:
            statement += f" The largest reported share was {top_percentage} percent."
        evidence.append(
            {
                "evidence_id": (
                    f"neon-holder:{network}:{target}:"
                    f"{_optional_text(row.get('holder_scanned_at')) or 'latest'}"
                ),
                "kind": "holder_snapshot",
                "statement": statement,
                "confidence": "VERIFIED",
                "rule_ids": [],
                "attributes": {
                    "source": "security_radar_holder_snapshots",
                    "holder_count": holder_count,
                    "top_percentage": top_percentage,
                },
            }
        )
        counts["holder"] += 1

    if not evidence:
        evidence.append(
            {
                "evidence_id": f"neon-summary:{source_id}",
                "kind": "scan_summary",
                "statement": _first_text(
                    [row.get("verdict"), "A signed unified radar verdict was recorded."]
                ),
                "confidence": "VERIFIED",
                "rule_ids": [],
                "attributes": {"source": "security_unified_radar_verdicts"},
            }
        )
        counts["fallback"] += 1

    record = {
        "schema_version": "arvis.export.v1",
        "case_id": f"neon-unified:{source_id}",
        "network": _kind(network, fallback="unknown-network"),
        "target": target,
        "signed_verdict": {
            "grade": grade,
            "signature": signature,
            "triggered_rules": list(dict.fromkeys(triggered_rule_ids)),
            "summary": _first_text(
                [row.get("verdict"), "A signed unified radar verdict was recorded."]
            ),
        },
        "evidence": evidence,
        "limitations": [
            "This training record contains only bounded evidence persisted by ARVIS."
        ],
        "lineage_ids": [f"{network}:{target}"],
    }
    return record, counts


def _validate_schema(rows: Iterable[Mapping[str, Any]]) -> None:
    available: dict[str, set[str]] = {}
    for row in rows:
        table = str(row.get("table_name") or "")
        column = str(row.get("column_name") or "")
        available.setdefault(table, set()).add(column)

    missing = []
    for table, required in sorted(_REQUIRED_COLUMNS.items()):
        absent = sorted(required - available.get(table, set()))
        if absent:
            missing.append(f"{table}: {', '.join(absent)}")
    if missing:
        raise ValueError("Neon source schema is incompatible: " + "; ".join(missing))


def _as_list(value: Any) -> list[Any]:
    parsed = _parse_json(value)
    return list(parsed) if isinstance(parsed, list) else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    parsed = _parse_json(value)
    return parsed if isinstance(parsed, Mapping) else {}


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _normalized_rule_id(value: Any) -> str | None:
    candidate = str(value or "").strip().upper()
    return candidate if _RULE_ID.fullmatch(candidate) else None


def _confidence_from_status(status: str) -> str:
    if status == "verified":
        return "VERIFIED"
    if status in {"observed", "inferred"}:
        return "INFERRED"
    return "UNVERIFIED"


def _kind(value: Any, *, fallback: str) -> str:
    candidate = _KIND_CHARS.sub("-", str(value or "").strip().lower()).strip("-._")
    return (candidate or fallback)[:64]


def _required_text(value: Any, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"Neon source row is missing {field}")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:4000] if text else None


def _first_text(values: Iterable[Any]) -> str:
    for value in values:
        text = _optional_text(value)
        if text is not None:
            return text
    raise ValueError("Neon source could not produce required text")


def _text_items(values: Iterable[Any]) -> list[str]:
    output = []
    for value in values:
        text = _optional_text(value)
        if text is not None:
            output.append(text)
    return output


def _primitive_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _primitive_number(value: Any) -> int | float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else round(number, 8)
