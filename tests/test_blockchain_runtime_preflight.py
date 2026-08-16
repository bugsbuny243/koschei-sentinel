from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.blockchain_base_candidates import (
    BlockchainBaseCandidate,
    LicenseReviewStatus,
    PreflightStatus,
    RuntimeIntegrationStatus,
    build_base_candidate_registry,
    load_base_candidate_registry,
)
from koschei_sentinel.blockchain_runtime_preflight import (
    BlockchainRuntimeProbeReceipt,
    audit_runtime_preflight,
    build_hardware_inventory,
    estimate_requirements,
    load_hardware_inventory,
    load_runtime_preflight_policy,
    plan_runtime_preflight,
    write_model,
)
from koschei_sentinel.training import model_digest


def _candidate(
    candidate_id: str = "native-base",
    *,
    total_billion: float = 16.0,
    license_status: LicenseReviewStatus = LicenseReviewStatus.ALLOWLISTED_OPEN_LICENSE,
    remote_code: bool = False,
) -> BlockchainBaseCandidate:
    blockers = ["runtime_preflight_not_run", "hardware_plan_not_approved"]
    runtime = RuntimeIntegrationStatus.NATIVE_TRANSFORMERS_EXPECTED
    if license_status is LicenseReviewStatus.MANUAL_REVIEW_REQUIRED:
        blockers.insert(0, "license_review_pending")
    if remote_code:
        blockers.insert(0, "remote_code_review_required")
        runtime = RuntimeIntegrationStatus.CUSTOM_CODE_REVIEW_REQUIRED
    return BlockchainBaseCandidate(
        candidate_id=candidate_id,
        lane="FEASIBILITY",
        model_id=f"fixture/{candidate_id}",
        revision="a" * 40,
        declared_total_parameters_billion=total_billion,
        declared_active_parameters_billion=min(2.0, total_billion),
        declared_context_length_tokens=131072,
        license_id="apache-2.0",
        license_review_status=license_status,
        commercial_use_declared=True,
        trust_remote_code_required=remote_code,
        runtime_integration_status=runtime,
        preflight_status=PreflightStatus.NOT_RUN,
        hardware_plan_approved=False,
        training_authorized=False,
        blocked_reasons=blockers,
    )


def _inventory(vram_mb: int = 24576) -> object:
    return build_hardware_inventory(
        inventory_id="fixture-gpu",
        cuda_available=True,
        gpu_count=1,
        gpu_model="Fixture GPU",
        vram_per_gpu_mb=vram_mb,
        host_ram_mb=262144,
        free_disk_mb=524288,
        compute_capability="8.0",
        bfloat16_supported=True,
    )


def _policy(root: Path):
    return load_runtime_preflight_policy(
        root / "configs/models/blockchain-runtime-preflight-policy.v1.json"
    )


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def test_preflight_resolves_only_preflight_transitional_blocks() -> None:
    root = Path(__file__).parents[1]
    candidate = _candidate()
    registry = build_base_candidate_registry("fixture-registry", [candidate])
    plan = plan_runtime_preflight(registry, candidate.candidate_id, _inventory(), _policy(root))

    assert plan.static_hardware_fit is True
    assert plan.runtime_probe_authorized is True
    assert plan.blockers == []
    assert plan.training_started is False
    assert plan.training_authorized is False


def test_remote_code_and_manual_license_remain_fail_closed() -> None:
    root = Path(__file__).parents[1]
    candidate = _candidate(
        "custom-base",
        license_status=LicenseReviewStatus.MANUAL_REVIEW_REQUIRED,
        remote_code=True,
    )
    registry = build_base_candidate_registry("fixture-registry", [candidate])
    plan = plan_runtime_preflight(registry, candidate.candidate_id, _inventory(), _policy(root))

    assert plan.runtime_probe_authorized is False
    assert "license_not_allowlisted" in plan.blockers
    assert "native_transformers_not_approved" in plan.blockers
    assert "trust_remote_code_required" in plan.blockers
    assert "candidate_block:license_review_pending" in plan.blockers
    assert "candidate_block:remote_code_review_required" in plan.blockers


def test_requirement_estimator_uses_total_parameter_footprint() -> None:
    root = Path(__file__).parents[1]
    policy = _policy(root)
    small = _candidate("small-base", total_billion=16.0)
    large = _candidate("large-base", total_billion=80.0)

    small_requirements = estimate_requirements(small, policy)
    large_requirements = estimate_requirements(large, policy)

    assert small_requirements[0] < large_requirements[0]
    assert small_requirements[2] < large_requirements[2]
    assert large_requirements[2] > 50000


def test_large_candidate_is_blocked_before_model_download_on_small_gpu() -> None:
    root = Path(__file__).parents[1]
    candidate = _candidate("large-base", total_billion=80.0)
    registry = build_base_candidate_registry("fixture-registry", [candidate])
    plan = plan_runtime_preflight(registry, candidate.candidate_id, _inventory(15360), _policy(root))

    assert plan.runtime_probe_authorized is False
    assert "insufficient_single_gpu_vram" in plan.blockers
    assert plan.estimated_min_vram_mb > plan.available_single_gpu_vram_mb


def test_hardware_inventory_digest_tampering_is_rejected(tmp_path: Path) -> None:
    inventory = _inventory()
    payload = inventory.model_dump(mode="json")
    payload["vram_per_gpu_mb"] += 1
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hardware inventory"):
        load_hardware_inventory(path)


def test_passed_probe_receipt_can_be_audited_without_raw_outputs() -> None:
    root = Path(__file__).parents[1]
    candidate = _candidate()
    registry = build_base_candidate_registry("fixture-registry", [candidate])
    inventory = _inventory()
    policy = _policy(root)
    plan = plan_runtime_preflight(registry, candidate.candidate_id, inventory, policy)
    payload = {
        "schema_version": "sentinel.blockchain-runtime-probe-receipt.v1",
        "candidate_id": candidate.candidate_id,
        "model_id": candidate.model_id,
        "revision": candidate.revision,
        "registry_digest": registry.registry_digest,
        "candidate_digest": model_digest(candidate),
        "policy_digest": model_digest(policy),
        "inventory_digest": inventory.inventory_digest,
        "plan_digest": plan.plan_digest,
        "environment_digest": "b" * 64,
        "tokenizer_loaded": True,
        "config_loaded": True,
        "quantized_model_loaded": True,
        "lora_attached": True,
        "forward_ok": True,
        "backward_ok": True,
        "exact_revision_observed": True,
        "trust_remote_code_used": False,
        "raw_model_outputs_stored": False,
        "probe_input_tokens": 64,
        "trainable_parameters": 1024,
        "total_parameters": 1_000_000,
        "peak_gpu_memory_mb": 12000,
        "status": "PASSED",
        "error_code": None,
        "training_started": False,
        "training_authorized": False,
    }
    receipt = BlockchainRuntimeProbeReceipt.model_validate(
        {**payload, "receipt_digest": _digest(payload)}
    )

    audit = audit_runtime_preflight(registry, inventory, policy, plan, receipt)

    assert audit.ready_for_training_authorization_review is True
    assert audit.runtime_probe_passed is True
    assert audit.headroom_mb == inventory.vram_per_gpu_mb - 12000
    assert audit.production_authority is False


def test_preflight_artifacts_are_no_replace(tmp_path: Path) -> None:
    inventory = _inventory()
    output = tmp_path / "inventory.json"
    write_model(inventory, output)

    with pytest.raises(FileExistsError):
        write_model(inventory, output)


def test_repository_shortlist_is_fail_closed_on_t4_class_inventory() -> None:
    root = Path(__file__).parents[1]
    registry = load_base_candidate_registry(
        root / "configs/models/blockchain-base-candidates.v1.json"
    )
    policy = _policy(root)
    t4 = build_hardware_inventory(
        inventory_id="t4-class",
        cuda_available=True,
        gpu_count=1,
        gpu_model="T4-class",
        vram_per_gpu_mb=15360,
        host_ram_mb=131072,
        free_disk_mb=524288,
        compute_capability="7.5",
        bfloat16_supported=False,
    )
    plans = {
        item.candidate_id: plan_runtime_preflight(registry, item.candidate_id, t4, policy)
        for item in registry.candidates
    }

    assert plans["deepseek-coder-v2-lite-base"].runtime_probe_authorized is False
    assert "trust_remote_code_required" in plans["deepseek-coder-v2-lite-base"].blockers
    assert plans["qwen3-coder-next-base"].runtime_probe_authorized is False
    assert "insufficient_single_gpu_vram" in plans["qwen3-coder-next-base"].blockers
    assert plans["glm-4.5-air-base"].runtime_probe_authorized is False
    assert "insufficient_single_gpu_vram" in plans["glm-4.5-air-base"].blockers
