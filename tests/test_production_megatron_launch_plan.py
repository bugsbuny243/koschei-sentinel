from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_megatron_launch_plan import (
    ProductionMegatronLaunchPlan,
    compile_production_megatron_launch_plan,
)
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def test_compile_candidate_preserves_397b_35b_budget_and_parallel_shape() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    assert plan.estimated_total_parameters_billion == pytest.approx(397.417644)
    assert plan.estimated_active_parameters_billion == pytest.approx(35.029778)
    assert plan.world_size == 1024
    assert plan.layers_per_pipeline_stage == 15
    assert plan.experts_per_ep_rank == 8
    assert plan.launchable is False
    assert plan.status == "dry_run_only"


def test_compiler_emits_critical_megatron_shape_arguments() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    argv = plan.argv
    for flag in (
        "--num-layers",
        "--hidden-size",
        "--num-attention-heads",
        "--num-query-groups",
        "--num-experts",
        "--moe-router-topk",
        "--expert-model-parallel-size",
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--context-parallel-size",
        "--sequence-parallel",
    ):
        assert flag in argv


def test_dry_run_plan_cannot_claim_launch_authority() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    payload = plan.model_dump(mode="json")
    payload["launchable"] = True
    with pytest.raises(ValueError):
        ProductionMegatronLaunchPlan.model_validate(payload)


def test_plan_hash_rejects_argument_mutation() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    payload = plan.model_dump(mode="json")
    payload["argv"] = [*payload["argv"], "--unexpected-mutation"]
    with pytest.raises(ValueError, match="self-hash mismatch"):
        ProductionMegatronLaunchPlan.model_validate(payload)


def test_plan_hash_is_deterministic() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    first = compile_production_megatron_launch_plan(spec)
    second = compile_production_megatron_launch_plan(spec)
    assert second.plan_sha256 == first.plan_sha256
