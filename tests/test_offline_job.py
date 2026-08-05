from __future__ import annotations

import pytest
from pydantic import ValidationError

from koschei_sentinel.autotrain import AutotrainPlan
from koschei_sentinel.offline_job import (
    OfflineJobBlocked,
    OfflineTrainingJob,
    build_offline_training_job,
)
from koschei_sentinel.training import (
    TrainingConfig,
    TrainingPlan,
    TrainingSplit,
    model_digest,
)

_RUN_ID = "sentinel-offline-v1"


def _config() -> TrainingConfig:
    return TrainingConfig(
        run_id=_RUN_ID,
        base_model="koschei-fixture/model",
        base_revision="1" * 40,
        dataset_release="build/releases/sentinel-offline-v1",
        readiness_report="build/releases/sentinel-offline-v1/readiness-report.json",
        require_readiness=True,
        output_dir="build/training/sentinel-offline-v1",
    )


def _plan(config: TrainingConfig) -> TrainingPlan:
    split = TrainingSplit(
        path="build/releases/sentinel-offline-v1/train.jsonl",
        examples=100,
        groups=100,
        digest="2" * 64,
    )
    return TrainingPlan(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        dataset_manifest_digest="3" * 64,
        readiness_report_digest="4" * 64,
        training_config_digest=model_digest(config),
        splits={"train": split, "validation": split, "test": split},
        effective_batch_size=16,
        estimated_optimizer_steps=7,
        output_dir=config.output_dir,
        warnings=[],
    )


def _autotrain(plan: TrainingPlan) -> AutotrainPlan:
    return AutotrainPlan(
        decision="ready_for_offline_training",
        training_run_id=plan.run_id,
        policy_digest="5" * 64,
        training_plan_digest=model_digest(plan),
        dataset_manifest_digest=plan.dataset_manifest_digest,
        readiness_report_digest=plan.readiness_report_digest,
        reasons=[],
        warnings=[],
    )


def test_offline_job_is_manual_and_incubation_only() -> None:
    config = _config()
    plan = _plan(config)

    job = build_offline_training_job(
        _autotrain(plan),
        config,
        plan,
        config_path="configs/training/offline.json",
        training_plan_path="build/jobs/offline.training-plan.json",
    )

    assert job.state == "planned_offline"
    assert job.candidate_stage == "incubation_candidate"
    assert job.command == [
        "sentinel-train",
        "--config",
        "configs/training/offline.json",
        "--plan-output",
        "build/jobs/offline.training-plan.json",
        "--execute",
    ]
    assert job.secret_values_included is False
    assert job.automatic_dispatch_allowed is False
    assert job.automatic_promotion_allowed is False
    assert job.production_deployment_allowed is False
    assert len(job.job_digest) == 64


def test_training_plan_digest_mismatch_blocks_job() -> None:
    config = _config()
    plan = _plan(config)
    autotrain = _autotrain(plan).model_copy(update={"training_plan_digest": "0" * 64})

    with pytest.raises(OfflineJobBlocked, match="training plan digest"):
        build_offline_training_job(
            autotrain,
            config,
            plan,
            config_path="configs/training/offline.json",
            training_plan_path="build/jobs/offline.training-plan.json",
        )


def test_repository_path_escape_is_rejected() -> None:
    config = _config()
    plan = _plan(config)

    with pytest.raises(ValueError, match="repository root"):
        build_offline_training_job(
            _autotrain(plan),
            config,
            plan,
            config_path="../private/config.json",
            training_plan_path="build/jobs/offline.training-plan.json",
        )


def test_job_schema_cannot_enable_automatic_dispatch() -> None:
    config = _config()
    plan = _plan(config)
    job = build_offline_training_job(
        _autotrain(plan),
        config,
        plan,
        config_path="configs/training/offline.json",
        training_plan_path="build/jobs/offline.training-plan.json",
    )
    payload = job.model_dump(mode="json")
    payload["automatic_dispatch_allowed"] = True

    with pytest.raises(ValidationError):
        OfflineTrainingJob.model_validate(payload)
