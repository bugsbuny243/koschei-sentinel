from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.dataset import DatasetExample, export_record
from koschei_sentinel.split import SplitConfig, split_dataset

SALT = "unit-test-salt-with-minimum-length"
RAW_TARGETS = (
    "8KxV7JGnVKXHh4UauuMxoPjzD4hnvy9dPp7Z4hXqW2aB",
    "7Lvc7jgQy4A7tFnXbUnJ8wuDkNnEnLrTj6Y3hFrkNmPq",
    "6NfVgQVKjLq9o5SxYz4mD2vJ7uWbT3cP8rE1aHkM9sXa",
)


def source(case_id: str, target: str, lineage: str) -> dict:
    return {
        "schema_version": "arvis.export.v1",
        "case_id": case_id,
        "network": "solana-mainnet",
        "target": target,
        "signed_verdict": {
            "grade": "D",
            "signature": f"fixture-signature-{case_id}",
            "triggered_rules": ["KS-HOLDER-001"],
            "summary": "A deterministic condition was observed.",
        },
        "evidence": [
            {
                "evidence_id": f"evidence-{case_id}",
                "kind": "holder_intelligence",
                "statement": "A material holder concentration was observed.",
                "confidence": "VERIFIED",
                "rule_ids": ["KS-HOLDER-001"],
            }
        ],
        "lineage_ids": [lineage],
    }


def examples() -> list[DatasetExample]:
    records = [
        source("case-1", RAW_TARGETS[0], "shared-lineage"),
        source("case-2", RAW_TARGETS[1], "shared-lineage"),
        source("case-3", RAW_TARGETS[2], "independent-lineage"),
    ]
    return [export_record(item, salt=SALT) for item in records]


def test_split_is_deterministic_and_group_safe(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    config = SplitConfig(seed="test-seed", train_bps=6000, validation_bps=2000)
    first = split_dataset(examples(), output_dir=first_dir, config=config)
    second = split_dataset(examples(), output_dir=second_dir, config=config)
    assert first == second
    assert (first_dir / "train.jsonl").read_bytes() == (
        second_dir / "train.jsonl"
    ).read_bytes()

    locations: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for line in (first_dir / f"{split_name}.jsonl").read_text().splitlines():
            item = DatasetExample.model_validate_json(line)
            previous = locations.setdefault(item.group_ref, split_name)
            assert previous == split_name


def test_duplicate_examples_fail_before_writing(tmp_path: Path) -> None:
    item = examples()[0]
    output_dir = tmp_path / "release"
    with pytest.raises(ValueError, match="duplicate example_id"):
        split_dataset([item, item], output_dir=output_dir)
    assert not output_dir.exists()


def test_privacy_finding_fails_before_writing(tmp_path: Path) -> None:
    item = examples()[0]
    raw = item.model_dump(mode="json")
    raw["case"]["evidence"][0]["statement"] = f"Raw wallet {RAW_TARGETS[0]}"
    output_dir = tmp_path / "release"
    with pytest.raises(ValueError, match="privacy quality gate failed"):
        split_dataset([raw], output_dir=output_dir)
    assert not output_dir.exists()


def test_dry_run_writes_nothing_and_reports_tiny_dataset() -> None:
    manifest = split_dataset([examples()[0]], output_dir=None, dry_run=True)
    assert manifest.total_examples == 1
    assert manifest.warnings


def test_existing_release_directory_is_never_overwritten(tmp_path: Path) -> None:
    output_dir = tmp_path / "release"
    output_dir.mkdir()
    marker = output_dir / "do-not-touch"
    marker.write_text("safe")
    with pytest.raises(FileExistsError):
        split_dataset(examples(), output_dir=output_dir)
    assert marker.read_text() == "safe"
