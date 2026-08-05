from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import (
    TrainingConfig,
    TrainingPlan,
    atomic_write,
    load_training_config,
    model_digest,
    plan_training,
)


class AutotrainPolicy(StrictModel):
    schema_version: Literal["sentinel.autotrain-policy.v1"] = (
        "sentinel.autotrain-policy.v1"
    )
    integration_state: Literal["incubation_only"] = "incubation_only"
    candidate_stage: Literal["incubation_candidate"] = "incubation_candidate"
    require_ready_dataset: Literal[True] = True
    require_training_readiness_binding: Literal[True] = True
    max_training_warnings: int = Field(default=0, ge=0, le=100)
    automatic_training_execution: Literal[False] = False
    automatic_promotion: Literal[False] = False
    production_deployment: Literal[False] = False


class AutotrainPlan(StrictModel):
    schema_version: Literal["sentinel.autotrain-plan.v1"] = (
        "sentinel.autotrain-plan.v1"
    )
    decision: Literal["ready_for_offline_training", "blocked"]
    integration_state: Literal["incubation_only"] = "incubation_only"
    candidate_stage: Literal["incubation_candidate"] = "incubation_candidate"
    training_run_id: str
    policy_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_plan_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    readiness_report_digest: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    automatic_training_execution_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def load_autotrain_policy(path: str | Path) -> AutotrainPolicy:
    try:
        return AutotrainPolicy.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid autotrain policy") from exc


def build_autotrain_plan(
    policy: AutotrainPolicy,
    config: TrainingConfig,
    training_plan: TrainingPlan,
) -> AutotrainPlan:
    reasons: list[str] = []

    if not config.require_readiness:
        reasons.append("training config must require a passing readiness report")
    if config.readiness_report is None:
        reasons.append("training config is missing readiness_report")
    if training_plan.readiness_report_digest is None:
        reasons.append("training plan is not bound to a readiness report")
    if len(training_plan.warnings) > policy.max_training_warnings:
        reasons.append(
            "training plan warnings "
            f"{len(training_plan.warnings)} exceed allowed maximum "
            f"{policy.max_training_warnings}"
        )

    return AutotrainPlan(
        decision="blocked" if reasons else "ready_for_offline_training",
        training_run_id=training_plan.run_id,
        policy_digest=model_digest(policy),
        training_plan_digest=model_digest(training_plan),
        dataset_manifest_digest=training_plan.dataset_manifest_digest,
        readiness_report_digest=training_plan.readiness_report_digest,
        reasons=reasons,
        warnings=list(training_plan.warnings),
    )


def plan_autotrain(
    policy_path: str | Path,
    config_path: str | Path,
    *,
    root: str | Path = ".",
) -> AutotrainPlan:
    policy = load_autotrain_policy(policy_path)
    config = load_training_config(config_path)
    training_plan = plan_training(config, root=root)
    return build_autotrain_plan(policy, config, training_plan)


def write_autotrain_plan(plan: AutotrainPlan, path: str | Path) -> None:
    payload = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)
