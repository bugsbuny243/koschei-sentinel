from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.curriculum import REQUIRED_STAGE_IDS, load_curriculum_policy
from koschei_sentinel.curriculum_cli import main

ROOT = Path(__file__).parents[1]
POLICY = ROOT / "configs/curriculum/language-first.v1.json"


def test_committed_language_first_policy_is_valid() -> None:
    policy = load_curriculum_policy(POLICY)

    assert tuple(stage.id for stage in policy.stages) == REQUIRED_STAGE_IDS
    assert policy.runtime_integration is False
    assert policy.source_contract.commit_sha_required is True
    assert policy.source_contract.compiler_oracle_required is True


def test_policy_rejects_missing_model_authority_deny(tmp_path: Path) -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    payload["authority_boundary"]["model_must_never"].remove("grant_capability")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model authority deny set incomplete"):
        load_curriculum_policy(broken)


def test_policy_rejects_missing_kosch_safety_deny(tmp_path: Path) -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    payload["kosch_boundary"]["must_never"].remove("pass_failed_language_gate")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="KOSCH authority deny set incomplete"):
        load_curriculum_policy(broken)


def test_policy_rejects_security_stage_before_language_gate(tmp_path: Path) -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    payload["stages"][4], payload["stages"][5] = payload["stages"][5], payload["stages"][4]
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
    assert '"language_gate": "language_hard_gate"' in output
    assert '"runtime_integration": false' in output
