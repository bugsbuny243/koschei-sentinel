from __future__ import annotations

from pydantic import ValidationError

from koschei_sentinel.autotrain import AutotrainPolicy, build_autotrain_plan
from koschei_sentinel.training import TrainingConfig, TrainingPlan, TrainingSplit


def _config(*, require_readiness: bool = True) -> TrainingConfig:
    return TrainingConfig(
        run_id="sentinel-incubation-v1",
        base_model="koschei-fixture/model",
        base_revision="0" * 40,
        dataset_release="build/releases/sentinel-v1",
        readiness_report="build/releases/sentinel-v1/readiness-report.json",
        require_readiness=require_readiness,
        output_dir="build/training/sentinel-incubation-v1",
    )


def _training_plan(*, warnings: list[str] | None = None) -> TrainingPlan:
    split = TrainingSplit(
        path="build/releases/sentinel-v1/train.jsonl",
        examples=100,
        groups=100,
        digest="1" * 64,
    )
    return TrainingPlan(
        run_id="sentinel-incubation-v1",
        base_model="koschei-fixture/model",
        base_revision="0" * 40,
        dataset_manifest_digest="2" * 64,
        readiness_report_digest="3" * 64,
        training_config_digest="4" * 64,
        splits={"train": split, "validation": split, "test": split},
        effective_batch_size=16,
        estimated_optimizer_steps=7,
        output_dir="build/training/sentinel-incubation-v1",
        warnings=warnings or [],
    )


def test_autotrain_plan_allows_only_offline_incubation_candidate() -> None:
    plan = build_autotrain_plan(
        AutotrainPolicy(),
        _config(),
        _training_plan(),
    )

    assert plan.decision == "ready_for_offline_training"
    assert plan.integration_state == "incubation_only"
    assert plan.candidate_stage == "incubation_candidate"
    assert plan.automatic_training_execution_allowed is False
    assert plan.automatic_promotion_allowed is False
    assert plan.production_deployment_allowed is False
    assert plan.reasons == []


def test_autotrain_plan_blocks_missing_readiness_requirement() -> None:
    plan = build_autotrain_plan(
        AutotrainPolicy(),
        _config(require_readiness=False),
        _training_plan(),
    )

    assert plan.decision == "blocked"
    assert "must require a passing readiness report" in plan.reasons[0]


def test_autotrain_plan_blocks_excess_training_warnings() -> None:
    plan = build_autotrain_plan(
        AutotrainPolicy(max_training_warnings=0),
        _config(),
        _training_plan(warnings=["validation split is empty"]),
    )

    assert plan.decision == "blocked"
    assert "exceed allowed maximum" in plan.reasons[0]


def test_autotrain_policy_cannot_enable_production_or_self_promotion() -> None:
    for field in (
        "automatic_training_execution",
        "automatic_promotion",
        "production_deployment",
    ):
        try:
            AutotrainPolicy.model_validate({field: True})
        except ValidationError:
            continue
        raise AssertionError(f"{field} unexpectedly accepted true")
