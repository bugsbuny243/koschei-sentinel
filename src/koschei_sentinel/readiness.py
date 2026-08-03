from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.dataset import DatasetExample
from koschei_sentinel.models import StrictModel
from koschei_sentinel.split import QualityManifest

_SPLIT_NAMES = ("train", "validation", "test")


class ReadinessPolicy(StrictModel):
    schema_version: Literal["sentinel.readiness-policy.v1"] = (
        "sentinel.readiness-policy.v1"
    )
    min_total_examples: int = Field(default=100, ge=1, le=10_000_000)
    min_total_groups: int = Field(default=50, ge=1, le=10_000_000)
    min_train_examples: int = Field(default=80, ge=1, le=10_000_000)
    min_validation_examples: int = Field(default=10, ge=0, le=10_000_000)
    min_test_examples: int = Field(default=10, ge=0, le=10_000_000)
    min_distinct_grades: int = Field(default=3, ge=1, le=6)
    min_distinct_evidence_kinds: int = Field(default=3, ge=1, le=512)
    max_largest_group_share_bps: int = Field(default=1000, ge=1, le=10_000)
    require_verified_evidence: bool = True
    required_grades: list[str] = Field(default_factory=list, max_length=6)
    required_evidence_kinds: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def requirements_are_consistent(self) -> ReadinessPolicy:
        if len(self.required_grades) != len(set(self.required_grades)):
            raise ValueError("required_grades must be unique")
        if len(self.required_evidence_kinds) != len(set(self.required_evidence_kinds)):
            raise ValueError("required_evidence_kinds must be unique")
        if any(value not in {"A", "B", "C", "D", "F", "-"} for value in self.required_grades):
            raise ValueError("required_grades contains an unsupported grade")
        if any(not value or len(value) > 64 for value in self.required_evidence_kinds):
            raise ValueError("required_evidence_kinds contains an invalid value")
        return self


class DatasetReadinessReport(StrictModel):
    schema_version: Literal["sentinel.dataset-readiness.v1"] = (
        "sentinel.dataset-readiness.v1"
    )
    ready: bool
    release_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    total_examples: int
    total_groups: int
    split_examples: dict[str, int]
    split_groups: dict[str, int]
    grades: list[str]
    evidence_kinds: list[str]
    confidence_labels: list[str]
    largest_group_examples: int
    largest_group_share_bps: int
    reasons: list[str] = Field(default_factory=list)
    policy: ReadinessPolicy


def load_readiness_policy(path: str | Path) -> ReadinessPolicy:
    try:
        return ReadinessPolicy.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid dataset readiness policy") from exc


def evaluate_release_readiness(
    release_dir: str | Path,
    *,
    policy: ReadinessPolicy | None = None,
) -> DatasetReadinessReport:
    active_policy = policy or ReadinessPolicy()
    release = Path(release_dir)
    manifest_path = release / "quality-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("dataset release is missing quality-manifest.json")

    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = QualityManifest.model_validate_json(manifest_bytes)
    except ValueError as exc:
        raise ValueError("invalid dataset quality manifest") from exc
    if manifest.dry_run:
        raise ValueError("readiness requires a materialized non-dry-run release")

    rows_by_split: dict[str, list[DatasetExample]] = {}
    for split_name in _SPLIT_NAMES:
        split_path = release / f"{split_name}.jsonl"
        if not split_path.is_file():
            raise ValueError(f"dataset release is missing {split_name}.jsonl")
        raw = split_path.read_bytes()
        report = manifest.splits.get(split_name)
        if report is None:
            raise ValueError(f"dataset manifest is missing {split_name} split")
        if hashlib.sha256(raw).hexdigest() != report.digest:
            raise ValueError(f"{split_name} split digest does not match quality manifest")
        rows = _parse_rows(raw, split_name)
        if len(rows) != report.examples:
            raise ValueError(f"{split_name} split count does not match quality manifest")
        if len({item.group_ref for item in rows}) != report.groups:
            raise ValueError(f"{split_name} group count does not match quality manifest")
        rows_by_split[split_name] = rows

    all_rows = [item for split_name in _SPLIT_NAMES for item in rows_by_split[split_name]]
    group_counts = Counter(item.group_ref for item in all_rows)
    split_membership: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in rows_by_split.items():
        for item in rows:
            split_membership[item.group_ref].add(split_name)

    reasons: list[str] = []
    crossed = sorted(group for group, names in split_membership.items() if len(names) > 1)
    if crossed:
        reasons.append(f"{len(crossed)} lineage groups cross dataset splits")

    total_examples = len(all_rows)
    total_groups = len(group_counts)
    if total_examples != manifest.total_examples:
        raise ValueError("dataset total example count does not match quality manifest")
    if total_groups != manifest.total_groups:
        raise ValueError("dataset total group count does not match quality manifest")

    split_examples = {name: len(rows_by_split[name]) for name in _SPLIT_NAMES}
    split_groups = {
        name: len({item.group_ref for item in rows_by_split[name]})
        for name in _SPLIT_NAMES
    }
    grade_counts = Counter(item.case.signed_verdict.grade for item in all_rows)
    evidence_kind_counts = Counter(
        evidence.kind for item in all_rows for evidence in item.case.evidence
    )
    confidence_counts = Counter(
        evidence.confidence.value for item in all_rows for evidence in item.case.evidence
    )
    largest_group_examples = max(group_counts.values(), default=0)
    largest_group_share_bps = (
        (largest_group_examples * 10_000 + total_examples - 1) // total_examples
        if total_examples
        else 0
    )

    _append_minimum_reason(
        reasons,
        "total examples",
        total_examples,
        active_policy.min_total_examples,
    )
    _append_minimum_reason(
        reasons,
        "total lineage groups",
        total_groups,
        active_policy.min_total_groups,
    )
    _append_minimum_reason(
        reasons,
        "train examples",
        split_examples["train"],
        active_policy.min_train_examples,
    )
    _append_minimum_reason(
        reasons,
        "validation examples",
        split_examples["validation"],
        active_policy.min_validation_examples,
    )
    _append_minimum_reason(
        reasons,
        "test examples",
        split_examples["test"],
        active_policy.min_test_examples,
    )
    _append_minimum_reason(
        reasons,
        "distinct grades",
        len(grade_counts),
        active_policy.min_distinct_grades,
    )
    _append_minimum_reason(
        reasons,
        "distinct evidence kinds",
        len(evidence_kind_counts),
        active_policy.min_distinct_evidence_kinds,
    )
    if largest_group_share_bps > active_policy.max_largest_group_share_bps:
        reasons.append(
            "largest lineage group share "
            f"{largest_group_share_bps} bps exceeds "
            f"{active_policy.max_largest_group_share_bps} bps"
        )
    if active_policy.require_verified_evidence and not confidence_counts.get("VERIFIED"):
        reasons.append("release contains no VERIFIED evidence")

    missing_grades = sorted(set(active_policy.required_grades) - set(grade_counts))
    if missing_grades:
        reasons.append("missing required grades: " + ", ".join(missing_grades))
    missing_kinds = sorted(
        set(active_policy.required_evidence_kinds) - set(evidence_kind_counts)
    )
    if missing_kinds:
        reasons.append("missing required evidence kinds: " + ", ".join(missing_kinds))

    return DatasetReadinessReport(
        ready=not reasons,
        release_manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(),
        policy_digest=_model_digest(active_policy),
        total_examples=total_examples,
        total_groups=total_groups,
        split_examples=split_examples,
        split_groups=split_groups,
        grades=sorted(grade_counts),
        evidence_kinds=sorted(evidence_kind_counts),
        confidence_labels=sorted(confidence_counts),
        largest_group_examples=largest_group_examples,
        largest_group_share_bps=largest_group_share_bps,
        reasons=reasons,
        policy=active_policy,
    )


def write_readiness_report(report: DatasetReadinessReport, path: str | Path) -> None:
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parse_rows(raw: bytes, split_name: str) -> list[DatasetExample]:
    rows: list[DatasetExample] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{split_name} split is not valid UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(DatasetExample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid {split_name} dataset row at line {line_number}"
            ) from exc
    return rows


def _append_minimum_reason(
    reasons: list[str],
    label: str,
    actual: int,
    required: int,
) -> None:
    if actual < required:
        reasons.append(f"{label} {actual} is below required minimum {required}")


def _model_digest(model: StrictModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
