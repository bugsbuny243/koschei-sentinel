from __future__ import annotations

import importlib.util
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_sft_training import (
    CyberSFTConfig,
    CyberSFTPlan,
    plan_cyber_sft,
)
from koschei_sentinel.models import StrictModel


class CyberTrainingUseClass(StrEnum):
    SMOKE_ONLY = "SMOKE_ONLY"
    PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CyberTrainingReadinessReport(StrictModel):
    schema_version: Literal["sentinel.cyber-training-readiness-report.v1"] = (
        "sentinel.cyber-training-readiness-report.v1"
    )
    run_id: str
    stage: str
    execution_profile: str
    use_class: CyberTrainingUseClass
    static_plan_ready: bool
    runtime_checked: bool
    runtime_dependencies_ready: bool | None
    cuda_available: bool | None
    visible_cuda_memory_gb: float | None = Field(default=None, ge=0.0)
    minimum_cuda_memory_gb: float = Field(ge=0.0)
    ready_to_execute: bool
    blockers: list[str]
    warnings: list[str]
    plan: CyberSFTPlan | None = None


def _use_class(plan: CyberSFTPlan) -> CyberTrainingUseClass:
    if plan.corpus_promotion_eligible is True:
        return CyberTrainingUseClass.PROMOTION_ELIGIBLE
    if plan.corpus_promotion_eligible is False:
        return CyberTrainingUseClass.SMOKE_ONLY
    return CyberTrainingUseClass.NOT_APPLICABLE


def _runtime_dependency_names() -> list[str]:
    return ["torch", "datasets", "peft", "transformers", "bitsandbytes", "accelerate"]


def audit_cyber_training_readiness(
    config: CyberSFTConfig,
    *,
    root: str | Path = ".",
    check_runtime: bool = False,
) -> CyberTrainingReadinessReport:
    blockers: list[str] = []
    warnings: list[str] = []
    plan: CyberSFTPlan | None = None
    try:
        plan = plan_cyber_sft(config, root=root)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        blockers.append(str(exc))

    if plan is not None:
        warnings.extend(plan.warnings)
        if not plan.executable_with_current_trainer:
            blockers.append("configured execution profile is not supported by the current Cyber SFT trainer")

    dependencies_ready: bool | None = None
    cuda_available: bool | None = None
    cuda_memory: float | None = None
    if check_runtime:
        missing = [name for name in _runtime_dependency_names() if importlib.util.find_spec(name) is None]
        dependencies_ready = not missing
        if missing:
            blockers.append("missing training runtime packages: " + ", ".join(missing))
        if dependencies_ready:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if not cuda_available:
                blockers.append("CUDA is not available")
            else:
                cuda_memory = sum(
                    torch.cuda.get_device_properties(index).total_memory
                    for index in range(torch.cuda.device_count())
                ) / (1024**3)
                if cuda_memory + 1e-9 < config.minimum_cuda_memory_gb:
                    blockers.append(
                        "visible CUDA memory below configured minimum: "
                        f"{cuda_memory:.1f} GiB < {config.minimum_cuda_memory_gb:.1f} GiB"
                    )
                if (
                    config.quantization.compute_dtype == "bfloat16"
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and not torch.cuda.is_bf16_supported()
                ):
                    blockers.append("configured bfloat16 is not supported by visible CUDA hardware")
    else:
        warnings.append("runtime packages and CUDA were not checked; use check_runtime on the GPU host")

    static_ready = plan is not None
    ready = static_ready and not blockers and (not check_runtime or dependencies_ready is True)
    return CyberTrainingReadinessReport(
        run_id=config.run_id,
        stage=config.stage.value,
        execution_profile=config.execution_profile.value,
        use_class=_use_class(plan) if plan is not None else CyberTrainingUseClass.NOT_APPLICABLE,
        static_plan_ready=static_ready,
        runtime_checked=check_runtime,
        runtime_dependencies_ready=dependencies_ready,
        cuda_available=cuda_available,
        visible_cuda_memory_gb=cuda_memory,
        minimum_cuda_memory_gb=config.minimum_cuda_memory_gb,
        ready_to_execute=ready,
        blockers=blockers,
        warnings=warnings,
        plan=plan,
    )
