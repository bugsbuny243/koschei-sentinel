from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from koschei_sentinel.autotrain import AutotrainPlan
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import (
    TrainingConfig,
    TrainingPlan,
    atomic_write,
    model_digest,
)

_DIGEST = r"^[a-f0-9]{64}$"
_RUN_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class OfflineJobBlocked(ValueError):
    """Raised when an artifact chain cannot produce an offline worker job."""


class OfflineTrainingJob(StrictModel):
    schema_version: Literal["sentinel.offline-training-job.v1"] = (
        "sentinel.offline-training-job.v1"
    )
    job_id: str = Field(pattern=_RUN_ID)
    state: Literal["planned_offline"] = "planned_offline"
    candidate_stage: Literal["incubation_candidate"] = "incubation_candidate"
    training_run_id: str = Field(pattern=_RUN_ID)
    training_config_path: str
    training_plan_path: str
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    dataset_manifest_digest: str = Field(pattern=_DIGEST)
    readiness_report_digest: str = Field(pattern=_DIGEST)
    training_config_digest: str = Field(pattern=_DIGEST)
    training_plan_digest: str = Field(pattern=_DIGEST)
    autotrain_plan_digest: str = Field(pattern=_DIGEST)
    output_dir: str
    command: list[str] = Field(min_length=6, max_length=16)
    secret_values_included: Literal[False] = False
    automatic_dispatch_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    job_digest: str = Field(pattern=_DIGEST)


def load_autotrain_plan(path: str | Path) -> AutotrainPlan:
    try:
        return AutotrainPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("invalid autotrain plan") from exc


def build_offline_training_job(
    autotrain: AutotrainPlan,
    config: TrainingConfig,
    plan: TrainingPlan,
    *,
    config_path: str,
    training_plan_path: str,
) -> OfflineTrainingJob:
    safe_config_path = _safe_relative_path(config_path, "training config path")
    safe_plan_path = _safe_relative_path(training_plan_path, "training plan path")

    if autotrain.decision != "ready_for_offline_training":
        raise OfflineJobBlocked("autotrain decision did not authorize offline training")
    if autotrain.training_run_id != config.run_id or config.run_id != plan.run_id:
        raise OfflineJobBlocked("training run IDs do not match")
    if autotrain.dataset_manifest_digest != plan.dataset_manifest_digest:
        raise OfflineJobBlocked("dataset digest does not match autotrain plan")
    if autotrain.readiness_report_digest is None:
        raise OfflineJobBlocked("autotrain plan is missing readiness binding")
    if plan.readiness_report_digest != autotrain.readiness_report_digest:
        raise OfflineJobBlocked("readiness digest does not match autotrain plan")
    if plan.training_config_digest != model_digest(config):
        raise OfflineJobBlocked("training config digest does not match training plan")
    if autotrain.training_plan_digest != model_digest(plan):
        raise OfflineJobBlocked("training plan digest does not match autotrain plan")
    if plan.output_dir != config.output_dir:
        raise OfflineJobBlocked("training output directory does not match config")

    command = [
        "sentinel-train",
        "--config",
        safe_config_path,
        "--plan-output",
        safe_plan_path,
        "--execute",
    ]
    payload = {
        "schema_version": "sentinel.offline-training-job.v1",
        "job_id": config.run_id,
        "state": "planned_offline",
        "candidate_stage": "incubation_candidate",
        "training_run_id": config.run_id,
        "training_config_path": safe_config_path,
        "training_plan_path": safe_plan_path,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "dataset_manifest_digest": plan.dataset_manifest_digest,
        "readiness_report_digest": autotrain.readiness_report_digest,
        "training_config_digest": plan.training_config_digest,
        "training_plan_digest": autotrain.training_plan_digest,
        "autotrain_plan_digest": model_digest(autotrain),
        "output_dir": config.output_dir,
        "command": command,
        "secret_values_included": False,
        "automatic_dispatch_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
    }
    return OfflineTrainingJob.model_validate(
        {**payload, "job_digest": _digest(payload)}
    )


def write_offline_training_job(job: OfflineTrainingJob, path: str | Path) -> None:
    payload = json.dumps(job.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)


def _safe_relative_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(f"{label} must stay within the repository root")
    return path.as_posix()


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
