from __future__ import annotations

from pathlib import Path

from koschei_sentinel.blockchain_base_candidates import load_base_candidate_registry
from koschei_sentinel.blockchain_runtime_preflight import (
    build_hardware_inventory,
    load_runtime_preflight_policy,
    plan_runtime_preflight,
)


def test_qwen25_coder_7b_is_the_native_t4_feasibility_lane() -> None:
    root = Path(__file__).parents[1]
    registry = load_base_candidate_registry(
        root / "configs/models/blockchain-base-candidates.v1.json"
    )
    policy = load_runtime_preflight_policy(
        root / "configs/models/blockchain-runtime-preflight-policy.v1.json"
    )
    hardware = build_hardware_inventory(
        inventory_id="t4-class",
        cuda_available=True,
        gpu_count=1,
        gpu_model="T4-class",
        vram_per_gpu_mb=15360,
        host_ram_mb=131072,
        free_disk_mb=262144,
        compute_capability="7.5",
        bfloat16_supported=False,
    )

    plan = plan_runtime_preflight(
        registry,
        "qwen2.5-coder-7b-base",
        hardware,
        policy,
    )

    assert plan.model_id == "Qwen/Qwen2.5-Coder-7B"
    assert plan.revision == "0396a76181e127dfc13e5c5ec48a8cee09938b02"
    assert plan.static_hardware_fit is True
    assert plan.runtime_probe_authorized is True
    assert plan.blockers == []
    assert plan.estimated_min_vram_mb < hardware.vram_per_gpu_mb
    assert plan.training_started is False
    assert plan.training_authorized is False
