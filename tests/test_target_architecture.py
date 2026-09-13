from __future__ import annotations

import pytest
from pydantic import ValidationError

from koschei_sentinel.target_architecture import TargetArchitectureContract


def test_default_contract_separates_a17b_validation_lane_from_35b_target() -> None:
    contract = TargetArchitectureContract()
    assert contract.target_total_parameters == "397B"
    assert contract.target_active_parameters == "35B"
    assert contract.current_executable_model == "Qwen/Qwen3.5-397B-A17B"
    assert contract.current_executable_active_parameters == "17B"
    assert contract.production_target_training_allowed is False
    assert contract.blockers


def test_contract_cannot_claim_current_lane_is_35b_active() -> None:
    with pytest.raises(ValidationError):
        TargetArchitectureContract(current_executable_active_parameters="35B")  # type: ignore[arg-type]


def test_contract_cannot_enable_production_target_training_early() -> None:
    with pytest.raises(ValidationError):
        TargetArchitectureContract(production_target_training_allowed=True)  # type: ignore[arg-type]
