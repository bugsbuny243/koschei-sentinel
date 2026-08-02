from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from koschei_sentinel.anonymize import (
    anonymize_record,
    pseudonymize,
    require_dataset_salt,
    sanitize_text,
)
from koschei_sentinel.models import (
    EvidenceConfidence,
    EvidenceItem,
    SecurityCase,
    SignedVerdict,
    StrictModel,
)

Primitive = str | int | float | bool | None
_RULE_ID = re.compile(r"^[A-Z][A-Z0-9._:-]{0,127}$")


class SourceEvidence(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=512)
    kind: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    statement: str = Field(min_length=1, max_length=4000)
    confidence: EvidenceConfidence
    rule_ids: list[str] = Field(default_factory=list, max_length=64)
    attributes: dict[str, Primitive] = Field(default_factory=dict, max_length=128)


class SourceSignedVerdict(StrictModel):
    grade: str = Field(pattern=r"^(A|B|C|D|F|-)$")
    signature: str = Field(min_length=8, max_length=512)
    triggered_rules: list[str] = Field(default_factory=list, max_length=128)
    summary: str = Field(min_length=1, max_length=4000)


class ARVISExportRecord(StrictModel):
    schema_version: Literal["arvis.export.v1"] = "arvis.export.v1"
    case_id: str = Field(min_length=1, max_length=512)
    network: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    target: str = Field(min_length=1, max_length=512)
    signed_verdict: SourceSignedVerdict
    evidence: list[SourceEvidence] = Field(min_length=1, max_length=512)
    limitations: list[str] = Field(default_factory=list, max_length=128)
    lineage_ids: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def source_evidence_ids_are_unique(self) -> ARVISExportRecord:
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("source evidence_id values must be unique")
        return self


class DatasetExample(StrictModel):
    schema_version: Literal["sentinel.dataset.v1"] = "sentinel.dataset.v1"
    example_id: str
    group_ref: str
    source_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    case: SecurityCase


class RejectedRecord(StrictModel):
    index: int
    source_digest: str
    error: str


class ExportManifest(StrictModel):
    schema_version: Literal["sentinel.export-manifest.v1"] = "sentinel.export-manifest.v1"
    dry_run: bool
    input_records: int
    accepted_records: int
    rejected_records: int
    output_digest: str | None = None
    examples: list[str] = Field(default_factory=list)
    rejected: list[RejectedRecord] = Field(default_factory=list)


def export_record(raw: Mapping[str, Any], *, salt: str | None = None) -> DatasetExample:
    active_salt = require_dataset_salt(salt)
    source = ARVISExportRecord.model_validate(raw)
    source_digest = source_fingerprint(source.model_dump(mode="json"), salt=active_salt)

    evidence = [
        EvidenceItem(
            evidence_id=pseudonymize(item.evidence_id, active_salt, prefix="evidence"),
            kind=item.kind,
            statement=sanitize_text(item.statement, salt=active_salt),
            confidence=item.confidence,
            rule_ids=_clean_rule_ids(item.rule_ids),
            attributes={
                key: _sanitize_attribute(key, value, salt=active_salt)
                for key, value in sorted(item.attributes.items())
            },
        )
        for item in source.evidence
    ]

    group_material = source.lineage_ids or [source.target]
    group_ref = pseudonymize("\x1f".join(sorted(group_material)), active_salt, prefix="group")
    case_id = pseudonymize(source.case_id, active_salt, prefix="case")

    case = SecurityCase(
        case_id=case_id,
        target_ref=pseudonymize(source.target, active_salt, prefix="target"),
        network=source.network,
        signed_verdict=SignedVerdict(
            grade=source.signed_verdict.grade,
            signature=pseudonymize(
                source.signed_verdict.signature, active_salt, prefix="signature"
            ),
            triggered_rules=_clean_rule_ids(source.signed_verdict.triggered_rules),
            summary=sanitize_text(source.signed_verdict.summary, salt=active_salt),
        ),
        evidence=evidence,
        limitations=[sanitize_text(item, salt=active_salt) for item in source.limitations],
    )
    return DatasetExample(
        example_id=pseudonymize(source.case_id, active_salt, prefix="example"),
        group_ref=group_ref,
        source_digest=source_digest,
        case=case,
    )


def export_jsonl(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path | None,
    salt: str | None = None,
    dry_run: bool = False,
) -> ExportManifest:
    active_salt = require_dataset_salt(salt)
    accepted: list[DatasetExample] = []
    rejected: list[RejectedRecord] = []
    seen_example_ids: set[str] = set()
    total = 0

    for index, raw in enumerate(records):
        total += 1
        digest = source_fingerprint(raw, salt=active_salt)
        try:
            example = export_record(raw, salt=active_salt)
            if example.example_id in seen_example_ids:
                raise ValueError("duplicate example_id after pseudonymization")
            seen_example_ids.add(example.example_id)
            accepted.append(example)
        except (TypeError, ValueError) as exc:
            rejected.append(
                RejectedRecord(index=index, source_digest=digest, error=_safe_error(exc))
            )

    accepted.sort(key=lambda item: item.example_id)
    payload = "".join(
        json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for item in accepted
    )
    output_digest = hashlib.sha256(payload.encode()).hexdigest() if accepted else None

    manifest = ExportManifest(
        dry_run=dry_run,
        input_records=total,
        accepted_records=len(accepted),
        rejected_records=len(rejected),
        output_digest=output_digest,
        examples=[item.example_id for item in accepted],
        rejected=rejected,
    )

    if rejected:
        return manifest
    if not dry_run:
        if output_path is None:
            raise ValueError("output_path is required unless dry_run is enabled")
        _atomic_write(Path(output_path), payload)
    return manifest


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON input must contain an array")
        return [_ensure_mapping(item) for item in parsed]
    return [_ensure_mapping(json.loads(line)) for line in text.splitlines() if line.strip()]


def write_manifest(manifest: ExportManifest, path: str | Path) -> None:
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    _atomic_write(Path(path), payload)


def source_fingerprint(value: Any, *, salt: str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hmac.new(salt.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _ensure_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("each input record must be a JSON object")
    return dict(value)


def _clean_rule_ids(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        if not _RULE_ID.fullmatch(cleaned):
            raise ValueError("rule_id must use the controlled uppercase identifier format")
        output.append(cleaned)
    return list(dict.fromkeys(output))


def _sanitize_attribute(key: str, value: Primitive, *, salt: str) -> Primitive:
    return anonymize_record({key: value}, salt=salt)[key]


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for item in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in item["loc"])
            parts.append(f"{location}: {item['type']}")
        return "; ".join(parts)
    return str(exc)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
