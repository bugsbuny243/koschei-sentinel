from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from koschei_sentinel.production_megatron_model_spec import ProductionMegatronModelSpec
from koschei_sentinel.production_transformer_budget import estimate_transformer_budget


CONFIG = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def _payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_candidate_is_target_sized_and_parallelism_is_coherent() -> None:
    spec = ProductionMegatronModelSpec.model_validate(_payload())
    budget = estimate_transformer_budget(spec.transformer)

    assert budget.within_target_tolerance
    assert budget.total_parameters_billion == pytest.approx(397.417644)
    assert budget.active_parameters_billion == pytest.approx(35.029778)
    assert spec.layers_per_pipeline_stage == 15
    assert spec.experts_per_ep_rank == 8
    assert spec.topology.world_size == 1024
    assert spec.status == "research_candidate"


def test_gqa_ratio_cannot_drift_from_budget() -> None:
    payload = _payload()
    payload["num_query_groups"] = 8
    with pytest.raises(ValidationError, match="GQA query-group ratio"):
        ProductionMegatronModelSpec.model_validate(payload)


def test_router_topk_cannot_drift_from_active_parameter_budget() -> None:
    payload = _payload()
    payload["moe_router_topk"] = 4
    with pytest.raises(ValidationError, match="router top-k"):
        ProductionMegatronModelSpec.model_validate(payload)


def test_experts_must_shard_evenly_across_ep() -> None:
    payload = _payload()
    payload["topology"]["expert_model_parallel_size"] = 32
    payload["topology"]["nodes"] = 256
    # 128 experts / EP32 is coherent, so force a non-divisible expert count instead.
    payload["transformer"]["num_experts"] = 130
    with pytest.raises(ValidationError):
        ProductionMegatronModelSpec.model_validate(payload)


def test_pipeline_stage_depth_must_be_integral() -> None:
    payload = _payload()
    payload["transformer"]["num_layers"] = 61
    with pytest.raises(ValidationError):
        ProductionMegatronModelSpec.model_validate(payload)
