from __future__ import annotations

import pytest

from koschei_sentinel.production_transformer_budget import (
    TransformerBudgetSpec,
    estimate_transformer_budget,
)


def test_reference_candidate_hits_397b_35b_budget_tolerance() -> None:
    spec = TransformerBudgetSpec(
        num_layers=60,
        hidden_size=8192,
        vocab_size=151936,
        attention_kv_ratio=0.125,
        num_experts=128,
        experts_per_token=8,
        expert_intermediate_size=2048,
        shared_intermediate_size=384,
        tied_embeddings=True,
    )
    result = estimate_transformer_budget(spec)
    assert result.total_parameters_billion == pytest.approx(397.417644, abs=1e-6)
    assert result.active_parameters_billion == pytest.approx(35.029778, abs=1e-6)
    assert result.dense_shared_parameters_billion == pytest.approx(10.870587, abs=1e-6)
    assert result.routed_expert_parameters_billion == pytest.approx(386.547057, abs=1e-6)
    assert result.active_routed_expert_parameters_billion == pytest.approx(24.159191, abs=1e-6)
    assert result.within_target_tolerance is True
    assert result.blockers == []
    assert result.status == "research_candidate"


def test_budget_outside_target_fails_closed() -> None:
    spec = TransformerBudgetSpec(
        num_layers=60,
        hidden_size=8192,
        vocab_size=151936,
        attention_kv_ratio=0.125,
        num_experts=128,
        experts_per_token=8,
        expert_intermediate_size=1024,
        shared_intermediate_size=384,
    )
    result = estimate_transformer_budget(spec)
    assert result.within_target_tolerance is False
    assert result.blockers


def test_router_cannot_activate_entire_expert_pool() -> None:
    with pytest.raises(ValueError):
        TransformerBudgetSpec(
            num_layers=60,
            hidden_size=8192,
            vocab_size=151936,
            attention_kv_ratio=0.125,
            num_experts=8,
            experts_per_token=8,
            expert_intermediate_size=2048,
            shared_intermediate_size=384,
        )
