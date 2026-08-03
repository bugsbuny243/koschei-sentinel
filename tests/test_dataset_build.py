from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.dataset_build import build_dataset_release
from koschei_sentinel.dataset_build_cli import main
from koschei_sentinel.readiness import load_readiness_policy
from koschei_sentinel.split import SplitConfig

FIXTURES = Path(__file__).parents[1] / "fixtures"
SOURCE = FIXTURES / "arvis.source.safe.json"
POLICY = FIXTURES / "readiness/policy.fixture.json"
SALT = "unit-test-salt-with-minimum-length"


def test_builder_dry_run_validates_without_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTINEL_DATASET_SALT", SALT)
    output = tmp_path / "release"

    manifest = build_dataset_release(
        [SOURCE],
        output_dir=output,
        salt_version="test-v1",
        policy=load_readiness_policy(POLICY),
        split_config=SplitConfig(seed="sentinel-split-v1"),
        dry_run=True,
    )

    assert manifest.status == "ready"
    assert not manifest.materialized
    assert not output.exists()


def test_builder_materializes_only_ready_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTINEL_DATASET_SALT", SALT)
    output = tmp_path / "release"

    manifest = build_dataset_release(
        [SOURCE],
        output_dir=output,
        salt_version="test-v1",
        policy=load_readiness_policy(POLICY),
        split_config=SplitConfig(seed="sentinel-split-v1"),
    )

    assert manifest.status == "ready"
    assert manifest.materialized
    assert sorted(path.name for path in output.iterdir()) == [
        "build-manifest.json",
        "quality-manifest.json",
        "readiness-report.json",
        "test.jsonl",
        "train.jsonl",
        "validation.jsonl",
    ]


def test_builder_default_policy_refuses_tiny_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTINEL_DATASET_SALT", SALT)
    output = tmp_path / "release"

    manifest = build_dataset_release(
        [SOURCE],
        output_dir=output,
        salt_version="test-v1",
    )

    assert manifest.status == "not_ready"
    assert not manifest.materialized
    assert not output.exists()
    assert manifest.readiness is not None
    assert manifest.readiness.reasons


def test_builder_rejects_duplicate_source_records_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTINEL_DATASET_SALT", SALT)
    output = tmp_path / "release"

    manifest = build_dataset_release(
        [SOURCE, SOURCE],
        output_dir=output,
        salt_version="test-v1",
        policy=load_readiness_policy(POLICY),
    )

    assert manifest.status == "rejected"
    assert manifest.export_manifest.rejected_records == 1
    assert not output.exists()


def test_builder_cli_materializes_ready_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTINEL_DATASET_SALT", SALT)
    output = tmp_path / "release"

    exit_code = main(
        [
            "--input",
            str(SOURCE),
            "--output-dir",
            str(output),
            "--salt-version",
            "test-v1",
            "--policy",
            str(POLICY),
        ]
    )

    assert exit_code == 0
    assert output.is_dir()
