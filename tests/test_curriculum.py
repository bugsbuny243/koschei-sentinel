from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.curriculum import (
    REQUIRED_SEMANTIC_PLANES,
    REQUIRED_STAGE_IDS,
    load_curriculum_policy,
)
from koschei_sentinel.curriculum_cli import main

ROOT = Path(__file__).parents[1]
POLICY = ROOT / "configs/curriculum/language-first.v1.json"


def test_committed_language_first_policy_is_valid() -> None:
    policy = load_curriculum_policy(POLICY)

    assert policy.schema_ == "sentinel.curriculum.v2"
    assert tuple(stage.id for stage in policy.stages) == REQUIRED_STAGE_IDS
    assert tuple(policy.semantic_planes) == REQUIRED_SEMANTIC_PLANES
    assert policy.runtime_integration is False
    assert policy.source_contract.required_contract_generation == 2
    assert policy.source_contract.commit_sha_required is True
    assert policy.source_contract.compiler_runtime_oracle_required is True
    assert policy.source_contract.semantic_plane_required is True
    gate_stage = next(stage for stage in policy.stages if stage.id == "N5")
    assert gate_stage.promotion_gate == "language_hard_gate"


def test_policy_rejects_missing_model_authority_deny(tmp_path: Path) -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    payload["authority_boundary"]["model_must_never"].remove(
        "grant_capability_or_authority"
    )
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model authority deny set incomplete"):
        load_curriculum_policy(broken)


def test_policy_rejects_missing_semantic_plane_zero_tolerance(tmp_path: Path) -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    payload["language_hard_gate"]["zero_tolerance"].remove("semantic_plane_confusion")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="language hard gate missing zero-tolerance rules"):
        load_curriculum_policy(broken)


def test_policy_rejects_security_stage_before_language_gate(tmp_path: Path) -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    payload["stages"][8], payload["stages"][9] = payload["stages"][9], payload["stages"][8]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="curriculum stages must be ordered exactly"):
        load_curriculum_policy(broken)


def test_curriculum_cli_accepts_committed_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--config", str(POLICY)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"schema": "sentinel.curriculum.v2"' in output
    assert '"language_gate": "language_hard_gate"' in output
    assert '"language_gate_stage": "N5"' in output
    assert '"runtime_integration": false' in output
