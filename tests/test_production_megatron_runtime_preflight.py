from __future__ import annotations

from pathlib import Path

import pytest

from koschei_sentinel.production_megatron_launch_plan import compile_production_megatron_launch_plan
from koschei_sentinel.production_megatron_model_spec import load_production_megatron_model_spec
from koschei_sentinel.production_megatron_runtime_preflight import (
    MegatronRuntimeInventory,
    MegatronRuntimePreflight,
    verify_runtime_preflight,
)


CANDIDATE = Path("configs/training/production-megatron-model.397b-35b.candidate.json")


def _inventory(**overrides):
    payload = {
        "nodes": 128,
        "gpus_per_node": 8,
        "minimum_observed_gpu_memory_gib": 80,
        "cuda_available_on_all_nodes": True,
        "nccl_available_on_all_nodes": True,
        "transformer_engine_available_on_all_nodes": True,
        "megatron_core_available_on_all_nodes": True,
        "homogeneous_gpu_count": True,
        "high_speed_interconnect_verified": True,
    }
    payload.update(overrides)
    return MegatronRuntimeInventory.model_validate(payload)


def test_matching_runtime_inventory_is_ready_but_not_authorized() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    result = verify_runtime_preflight(spec, plan, _inventory())
    assert result.runtime_ready is True
    assert result.blockers == []
    assert result.launch_authorized is False
    assert result.required_world_size == 1024


def test_low_gpu_memory_fails_closed() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    result = verify_runtime_preflight(
        spec, plan, _inventory(minimum_observed_gpu_memory_gib=79)
    )
    assert result.runtime_ready is False
    assert "minimum observed GPU memory is below topology requirement" in result.blockers


def test_missing_interconnect_verification_fails_closed() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    result = verify_runtime_preflight(
        spec, plan, _inventory(high_speed_interconnect_verified=False)
    )
    assert result.runtime_ready is False
    assert "high-speed interconnect has not been verified" in result.blockers


def test_runtime_preflight_hash_rejects_tampering() -> None:
    spec = load_production_megatron_model_spec(CANDIDATE)
    plan = compile_production_megatron_launch_plan(spec)
    result = verify_runtime_preflight(spec, plan, _inventory())
    payload = result.model_dump(mode="json")
    payload["observed_world_size"] = 1
    with pytest.raises(ValueError, match="self-hash mismatch"):
        MegatronRuntimePreflight.model_validate(payload)
