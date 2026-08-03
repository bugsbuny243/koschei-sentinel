from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from koschei_sentinel.readiness import (
    ReadinessPolicy,
    evaluate_release_readiness,
    load_readiness_policy,
    write_readiness_report,
)
from koschei_sentinel.readiness_cli import main

FIXTURES = Path(__file__).parents[1] / "fixtures"
RELEASE = FIXTURES / "training/release"
POLICY = FIXTURES / "readiness/policy.fixture.json"


def test_default_policy_rejects_tiny_fixture() -> None:
    report = evaluate_release_readiness(RELEASE)

    assert not report.ready
    assert report.total_examples == 1
    assert report.total_groups == 1
    assert report.reasons


def test_fixture_policy_accepts_valid_release_and_writes_report(tmp_path: Path) -> None:
    policy = load_readiness_policy(POLICY)
    report = evaluate_release_readiness(RELEASE, policy=policy)
    output = tmp_path / "readiness.json"

    write_readiness_report(report, output)

    assert report.ready
    assert report.reasons == []
    assert report.split_examples == {"train": 1, "validation": 0, "test": 0}
    assert json.loads(output.read_text())["ready"] is True


def test_readiness_rejects_split_digest_drift(tmp_path: Path) -> None:
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    train = release / "train.jsonl"
    train.write_text(train.read_text() + "\n")

    with pytest.raises(ValueError, match="digest"):
        evaluate_release_readiness(release, policy=load_readiness_policy(POLICY))


def test_policy_rejects_duplicate_requirements() -> None:
    with pytest.raises(ValueError, match="required_grades"):
        ReadinessPolicy(required_grades=["D", "D"])


def test_readiness_cli_uses_distinct_not_ready_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report.json"
    exit_code = main(["--release", str(RELEASE), "--output", str(output)])

    assert exit_code == 3
    assert output.is_file()
    assert '"ready": false' in capsys.readouterr().out
