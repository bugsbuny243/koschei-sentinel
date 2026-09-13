from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from koschei_sentinel.production_target_candidate_matrix import (
    ProductionTargetCandidate,
    load_production_target_candidate_matrix,
)


def _candidate(**overrides):
    payload = {
        "model_ref": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "total_parameters_billion": 480.0,
        "active_parameters_billion": 35.0,
        "architecture": "qwen3_moe",
        "license": "apache-2.0",
        "framework": {
            "transformers": "supported",
            "ms_swift": "supported",
            "megatron_swift": "supported",
        },
        "fit": {
            "total_target_match": False,
            "active_target_match": True,
            "exact_target_match": False,
        },
        "role": "active-width-and-router-reference",
        "production_binding_allowed": False,
        "evidence": ["official model card"],
    }
    payload.update(overrides)
    return ProductionTargetCandidate.model_validate(payload)


def test_480b_a35b_is_reference_not_exact_target() -> None:
    candidate = _candidate()
    assert candidate.fit.active_target_match is True
    assert candidate.fit.total_target_match is False
    assert candidate.fit.exact_target_match is False
    assert candidate.production_binding_allowed is False


def test_397b_a17b_cannot_claim_active_match() -> None:
    with pytest.raises(ValidationError, match="active_target_match"):
        _candidate(
            model_ref="Qwen/Qwen3.5-397B-A17B",
            total_parameters_billion=397.0,
            active_parameters_billion=17.0,
            fit={
                "total_target_match": True,
                "active_target_match": True,
                "exact_target_match": True,
            },
        )


def test_matrix_file_remains_fail_closed() -> None:
    matrix = load_production_target_candidate_matrix(
        "configs/training/production-target-candidate-matrix.v1.json"
    )
    assert matrix.decision.exact_public_base_found is False
    assert matrix.decision.production_binding_allowed is False
    assert not any(candidate.fit.exact_target_match for candidate in matrix.candidates)
