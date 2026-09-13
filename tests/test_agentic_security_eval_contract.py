from pathlib import Path

import pytest

from koschei_sentinel.agentic_security_eval_contract import Config, load_and_validate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/agentic-security-protocol-state-evals.v1.json"
SEEDS = ROOT / "configs/training/agentic-security-protocol-state-seeds.v1.json"


def test_committed_agentic_security_bundle_is_structurally_valid() -> None:
    config, seeds = load_and_validate(CONFIG, SEEDS)
    assert config.target_architecture.total_parameters == "397B"
    assert config.target_architecture.active_parameters == "35B"
    assert {case.family for case in seeds.cases} == {family.id for family in config.families}


def test_valid_credential_shortcut_cannot_be_removed() -> None:
    config, _ = load_and_validate(CONFIG, SEEDS)
    payload = config.model_dump(mode="json")
    payload["forbidden_shortcuts"].remove("valid_credential_implies_correct_principal")
    with pytest.raises(ValueError, match="shortcuts cannot be removed"):
        Config.model_validate(payload)


def test_code_presence_cannot_be_treated_as_protocol_activation() -> None:
    config, _ = load_and_validate(CONFIG, SEEDS)
    payload = config.model_dump(mode="json")
    payload["forbidden_shortcuts"].remove("feature_present_in_code_implies_activated_on_chain")
    with pytest.raises(ValueError, match="shortcuts cannot be removed"):
        Config.model_validate(payload)
