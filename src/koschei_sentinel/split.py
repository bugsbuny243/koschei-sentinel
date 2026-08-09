from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.anonymize import detect_sensitive_text
from koschei_sentinel.dataset import DatasetExample
from koschei_sentinel.models import StrictModel

SplitName = Literal["train", "validation", "test"]
_PSEUDONYM = re.compile(r"^[a-z][a-z0-9_]*_[a-f0-9]{24}$")


class SplitConfig(StrictModel):
    seed: str = Field(default="sentinel-split-v1", min_length=1, max_length=128)
    train_bps: int = Field(default=8000, ge=1, le=9998)
    validation_bps: int = Field(default=1000, ge=1, le=9998)

    @model_validator(mode="after")
    def ratios_leave_room_for_test(self) -> SplitConfig:
        if self.train_bps + self.validation_bps >= 10000:
            raise ValueError("train_bps + validation_bps must be less than 10000")
        return self

    @property
    def test_bps(self) -> int:
        return 10000 - self.train_bps - self.validation_bps


class SplitReport(StrictModel):
    examples: int
    groups: int
    digest: str


class QualityManifest(StrictModel):
    schema_version: Literal["sentinel.quality-manifest.v1"] = "sentinel.quality-manifest.v1"
    dry_run: bool
    seed: str
    train_bps: int
    validation_bps: int
    test_bps: int
    total_examples: int
    total_groups: int
    splits: dict[str, SplitReport]
    grade_counts: dict[str, int]
    confidence_counts: dict[str, int]
    evidence_kind_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)


def split_dataset(
    examples: Iterable[DatasetExample | Mapping[str, Any]],
    *,
    output_dir: str | Path | None,
    config: SplitConfig | None = None,
    dry_run: bool = False,
) -> QualityManifest:
    active_config = config or SplitConfig()
    validated = [
        item if isinstance(item, DatasetExample) else DatasetExample.model_validate(item)
        for item in examples
    ]
    if not validated:
        raise ValueError("dataset must contain at least one example")

    _validate_uniqueness(validated)
    privacy_findings = _privacy_findings(validated)
    if privacy_findings:
        raise ValueError("privacy quality gate failed: " + "; ".join(privacy_findings))

    grouped = _lineage_components(validated)

    assigned: dict[SplitName, dict[str, list[DatasetExample]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    for component_ref, members in grouped.items():
        split_name = _assign_split(component_ref, active_config)
        assigned[split_name][component_ref] = members

    payloads: dict[SplitName, str] = {}
    reports: dict[str, SplitReport] = {}
    warnings: list[str] = []
    for split_name in ("train", "validation", "test"):
        members = sorted(
            (
                item
                for component_members in assigned[split_name].values()
                for item in component_members
            ),
            key=lambda item: item.example_id,
        )
        payload = "".join(
            _canonical_json(item.model_dump(mode="json")) + "\n" for item in members
        )
        payloads[split_name] = payload
        reports[split_name] = SplitReport(
            examples=len(members),
            groups=len(assigned[split_name]),
            digest=hashlib.sha256(payload.encode()).hexdigest(),
        )
        if not members:
            warnings.append(f"{split_name} split is empty")

    manifest = QualityManifest(
        dry_run=dry_run,
        seed=active_config.seed,
        train_bps=active_config.train_bps,
        validation_bps=active_config.validation_bps,
        test_bps=active_config.test_bps,
        total_examples=len(validated),
        total_groups=len(grouped),
        splits=reports,
        grade_counts=dict(
            sorted(Counter(item.case.signed_verdict.grade for item in validated).items())
        ),
        confidence_counts=dict(
            sorted(
                Counter(
                    evidence.confidence.value
                    for item in validated
                    for evidence in item.case.evidence
                ).items()
            )
        ),
        evidence_kind_counts=dict(
            sorted(
                Counter(
                    evidence.kind for item in validated for evidence in item.case.evidence
                ).items()
            )
        ),
        warnings=warnings,
    )

    if not dry_run:
        if output_dir is None:
            raise ValueError("output_dir is required unless dry_run is enabled")
        _write_release(Path(output_dir), payloads, manifest)
    return manifest


def load_dataset(path: str | Path) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            examples.append(DatasetExample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid dataset row at line {line_number}") from exc
    return examples


def _assign_split(group_ref: str, config: SplitConfig) -> SplitName:
    digest = hashlib.sha256(f"{config.seed}\x1f{group_ref}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    if bucket < config.train_bps:
        return "train"
    if bucket < config.train_bps + config.validation_bps:
        return "validation"
    return "test"


def _lineage_components(examples: list[DatasetExample]) -> dict[str, list[DatasetExample]]:
    """Collapse transitive lineage overlap so related families cannot cross splits."""

    parents = list(range(len(examples)))
    owners: dict[str, int] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for index, example in enumerate(examples):
        identities = [f"group:{example.group_ref}"]
        identities.extend(f"lineage:{item}" for item in example.lineage_refs)
        for identity in identities:
            previous = owners.get(identity)
            if previous is None:
                owners[identity] = index
            else:
                union(index, previous)

    components: dict[int, list[DatasetExample]] = defaultdict(list)
    for index, example in enumerate(examples):
        components[find(index)].append(example)

    result: dict[str, list[DatasetExample]] = {}
    for members in components.values():
        component_ref = min(
            [item.group_ref for item in members]
            + [lineage for item in members for lineage in item.lineage_refs]
        )
        result[component_ref] = members
    return result


def _validate_uniqueness(examples: list[DatasetExample]) -> None:
    example_ids = [item.example_id for item in examples]
    source_digests = [item.source_digest for item in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("duplicate example_id detected")
    if len(source_digests) != len(set(source_digests)):
        raise ValueError("duplicate source_digest detected")


def _privacy_findings(examples: list[DatasetExample]) -> list[str]:
    findings: list[str] = []
    for example in examples:
        identifiers = {
            "example_id": example.example_id,
            "group_ref": example.group_ref,
            "case_id": example.case.case_id,
            "target_ref": example.case.target_ref,
            "signature": example.case.signed_verdict.signature,
        }
        identifiers.update(
            {
                f"lineage_ref[{index}]": value
                for index, value in enumerate(example.lineage_refs)
            }
        )
        identifiers.update(
            {
                f"evidence_id[{index}]": item.evidence_id
                for index, item in enumerate(example.case.evidence)
            }
        )
        for field, value in identifiers.items():
            if not _PSEUDONYM.fullmatch(value):
                findings.append(f"{example.example_id}.{field}: identifier is not pseudonymized")

        text_values = [
            ("verdict.summary", example.case.signed_verdict.summary),
            *(
                (f"limitation[{index}]", value)
                for index, value in enumerate(example.case.limitations)
            ),
        ]
        for evidence_index, evidence in enumerate(example.case.evidence):
            text_values.append((f"evidence[{evidence_index}].statement", evidence.statement))
            text_values.extend(
                (f"evidence[{evidence_index}].attributes.{key}", value)
                for key, value in evidence.attributes.items()
                if isinstance(value, str)
            )
        for field, value in text_values:
            for finding in detect_sensitive_text(value):
                findings.append(f"{example.example_id}.{field}: {finding}")
    return findings


def _write_release(
    output_dir: Path,
    payloads: dict[SplitName, str],
    manifest: QualityManifest,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for split_name, payload in payloads.items():
            (staging / f"{split_name}.jsonl").write_text(payload, encoding="utf-8")
        (staging / "quality-manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_release(staging)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _fsync_release(staging: Path) -> None:
    for path in staging.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
