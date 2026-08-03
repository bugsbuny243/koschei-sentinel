from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.readiness import (
    evaluate_release_readiness,
    load_readiness_policy,
    write_readiness_report,
)
from koschei_sentinel.training import TrainingConfig, plan_training

FIXTURES = Path(__file__).parents[1] / "fixtures"
SOURCE_RELEASE = FIXTURES / "training/release"
POLICY = FIXTURES / "readiness/policy.fixture.json"


def _prepared_root(tmp_path: Path) -> Path:
    release = tmp_path / "build/releases/ready"
    release.parent.mkdir(parents=True)
    shutil.copytree(SOURCE_RELEASE, release)
    report = evaluate_release_readiness(release, policy=load_readiness_policy(POLICY))
    write_readiness_report(report, release / "readiness-report.json")
    return tmp_path


def _config() -> TrainingConfig:
    return TrainingConfig(
        run_id="ready-v0.7",
        base_model="sentinel-fixture/model",
        base_revision="0" * 40,
        dataset_release="build/releases/ready",
        readiness_report="build/releases/ready/readiness-report.json",
        require_readiness=True,
        output_dir="build/training/ready-v0.7",
    )


def test_training_plan_binds_passing_readiness_report(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)

    plan = plan_training(_config(), root=root)

    assert plan.readiness_report_digest is not None
    assert plan.warnings == ["validation split is empty", "test split is empty"]


def test_training_plan_rejects_readiness_report_manifest_mismatch(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path)
    report_path = root / "build/releases/ready/readiness-report.json"
    payload = json.loads(report_path.read_text())
    payload["release_manifest_digest"] = "f" * 64
    report_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match"):
        plan_training(_config(), root=root)


def test_training_config_requires_report_when_readiness_is_required() -> None:
    with pytest.raises(ValidationError, match="readiness_report"):
        TrainingConfig(
            run_id="missing-readiness",
            base_model="owner/model",
            base_revision="0" * 40,
            dataset_release="build/releases/ready",
            require_readiness=True,
            output_dir="build/training/missing-readiness",
        )
