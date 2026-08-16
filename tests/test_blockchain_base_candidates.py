from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.blockchain_base_candidates import (
    BlockchainBaseCandidate,
    BlockchainBaseCandidatePolicy,
    LicenseReviewStatus,
    PreflightStatus,
    RuntimeIntegrationStatus,
    audit_base_candidate_registry,
    build_base_candidate_registry,
    load_base_candidate_registry,
)


def _candidate(
    candidate_id: str,
    model_id: str,
    revision_character: str,
    lane: str,
    *,
    license_status: LicenseReviewStatus = LicenseReviewStatus.ALLOWLISTED_OPEN_LICENSE,
    remote_code: bool = False,
) -> BlockchainBaseCandidate:
    blocked = ["runtime_preflight_not_run", "hardware_plan_not_approved"]
    runtime_status = RuntimeIntegrationStatus.NATIVE_TRANSFORMERS_EXPECTED
    license_id = "apache-2.0"
    if license_status is LicenseReviewStatus.MANUAL_REVIEW_REQUIRED:
        blocked.insert(0, "license_review_pending")
        license_id = "custom-model-license"
    if remote_code:
        blocked.insert(1, "remote_code_review_required")
        runtime_status = RuntimeIntegrationStatus.CUSTOM_CODE_REVIEW_REQUIRED
    return BlockchainBaseCandidate(
        candidate_id=candidate_id,
        lane=lane,
        model_id=model_id,
        revision=revision_character * 40,
        declared_total_parameters_billion=16,
        declared_active_parameters_billion=2,
        declared_context_length_tokens=131072,
        license_id=license_id,
        license_review_status=license_status,
        commercial_use_declared=True,
        trust_remote_code_required=remote_code,
        runtime_integration_status=runtime_status,
        preflight_status=PreflightStatus.NOT_RUN,
        hardware_plan_approved=False,
        training_authorized=False,
        blocked_reasons=blocked,
    )


def _policy() -> BlockchainBaseCandidatePolicy:
    return BlockchainBaseCandidatePolicy(
        policy_id="candidate-test",
        min_candidates=3,
        required_lanes=["FEASIBILITY", "EFFICIENCY", "POWER"],
    )


def test_three_lane_registry_is_admitted_for_preflight_but_not_training() -> None:
    registry = build_base_candidate_registry(
        "candidate-test",
        [
            _candidate(
                "small-base",
                "fixture/small-base",
                "1",
                "FEASIBILITY",
                license_status=LicenseReviewStatus.MANUAL_REVIEW_REQUIRED,
                remote_code=True,
            ),
            _candidate("sparse-base", "fixture/sparse-base", "2", "EFFICIENCY"),
            _candidate("large-base", "fixture/large-base", "3", "POWER"),
        ],
    )

    audit = audit_base_candidate_registry(registry, _policy())

    assert audit.ready_for_preflight is True
    assert audit.admitted_for_preflight == 3
    assert audit.ready_for_training_authorization_review == 0
    assert audit.violations == []
    assert "small-base" in audit.blocked_candidates


def test_remote_code_cannot_bypass_custom_code_review() -> None:
    with pytest.raises(ValueError, match="remote-code"):
        BlockchainBaseCandidate(
            candidate_id="unsafe-base",
            lane="FEASIBILITY",
            model_id="fixture/unsafe-base",
            revision="a" * 40,
            declared_total_parameters_billion=16,
            declared_active_parameters_billion=2,
            license_id="apache-2.0",
            license_review_status="ALLOWLISTED_OPEN_LICENSE",
            commercial_use_declared=True,
            trust_remote_code_required=True,
            runtime_integration_status="NATIVE_TRANSFORMERS_EXPECTED",
            training_authorized=False,
            blocked_reasons=["runtime_preflight_not_run"],
        )


def test_passed_preflight_requires_hardware_plan() -> None:
    with pytest.raises(ValueError, match="hardware plan"):
        BlockchainBaseCandidate(
            candidate_id="preflight-base",
            lane="EFFICIENCY",
            model_id="fixture/preflight-base",
            revision="b" * 40,
            declared_total_parameters_billion=16,
            declared_active_parameters_billion=2,
            license_id="apache-2.0",
            license_review_status="ALLOWLISTED_OPEN_LICENSE",
            commercial_use_declared=True,
            trust_remote_code_required=False,
            runtime_integration_status="NATIVE_TRANSFORMERS_EXPECTED",
            preflight_status="PASSED",
            hardware_plan_approved=False,
            training_authorized=False,
        )


def test_registry_digest_tampering_is_rejected(tmp_path: Path) -> None:
    registry = build_base_candidate_registry(
        "candidate-test",
        [
            _candidate("one-base", "fixture/one-base", "1", "FEASIBILITY"),
            _candidate("two-base", "fixture/two-base", "2", "EFFICIENCY"),
            _candidate("three-base", "fixture/three-base", "3", "POWER"),
        ],
    )
    payload = registry.model_dump(mode="json")
    payload["candidates"][0]["declared_total_parameters_billion"] = 17
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registry"):
        load_base_candidate_registry(path)


def test_repository_registry_pins_reviewed_initial_shortlist() -> None:
    root = Path(__file__).parents[1]
    registry = load_base_candidate_registry(
        root / "configs/models/blockchain-base-candidates.v1.json"
    )
    policy = BlockchainBaseCandidatePolicy.model_validate_json(
        (root / "configs/models/blockchain-base-candidate-policy.v1.json").read_text(
            encoding="utf-8"
        )
    )
    audit = audit_base_candidate_registry(registry, policy)

    assert audit.ready_for_preflight is True
    assert audit.candidates == 4
    assert audit.ready_for_training_authorization_review == 0
    by_id = {item.candidate_id: item for item in registry.candidates}
    assert by_id["qwen2.5-coder-7b-base"].revision == (
        "0396a76181e127dfc13e5c5ec48a8cee09938b02"
    )
    assert by_id["qwen2.5-coder-7b-base"].declared_total_parameters_billion == 7.61
    assert by_id["qwen2.5-coder-7b-base"].license_review_status is (
        LicenseReviewStatus.ALLOWLISTED_OPEN_LICENSE
    )
    assert by_id["qwen2.5-coder-7b-base"].runtime_integration_status is (
        RuntimeIntegrationStatus.NATIVE_TRANSFORMERS_EXPECTED
    )
    assert by_id["qwen3-coder-next-base"].revision == (
        "1b6df59d5f75ab51edb9ad8cb3ea69c5d0aedd57"
    )
    assert by_id["deepseek-coder-v2-lite-base"].revision == (
        "5a3cf151e6eb71197e34b87d42546c97b2adb139"
    )
    assert by_id["glm-4.5-air-base"].revision == (
        "b52435f955c9dc2c38e75319e3d2f0bf511e15ff"
    )
    assert by_id["deepseek-coder-v2-lite-base"].license_review_status is (
        LicenseReviewStatus.MANUAL_REVIEW_REQUIRED
    )
    assert by_id["deepseek-coder-v2-lite-base"].runtime_integration_status is (
        RuntimeIntegrationStatus.CUSTOM_CODE_REVIEW_REQUIRED
    )
    assert all(item.training_authorized is False for item in registry.candidates)
