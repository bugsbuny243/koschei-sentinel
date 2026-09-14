from pathlib import Path

import pytest

from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_model_construction import build_construction_plan

CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_construction_plan_preserves_parallel_sharding() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_construction_plan(spec)
    assert plan.dense_parameter_shards == 32
    assert plan.expert_parameter_shards == 512
    assert plan.checkpoint_shards == 1024
    assert plan.execution_authorized is False
    assert plan.status == "dry_run_only"


def test_weight_memory_estimate_is_positive_and_bounded() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_construction_plan(spec)
    assert plan.estimated_dense_weight_gib_per_rank > 0
    assert plan.estimated_expert_weight_gib_per_rank > 0
    assert plan.estimated_model_weight_gib_per_rank < 80
    assert plan.estimated_optimizer_state_gib_per_rank_lower_bound > 0


def test_plan_keeps_unmeasured_runtime_memory_as_blocker() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = build_construction_plan(spec)
    assert plan.activation_memory_estimated is False
    assert any("activation memory" in blocker for blocker in plan.blockers)
    assert any("optimizer step" in blocker for blocker in plan.blockers)
