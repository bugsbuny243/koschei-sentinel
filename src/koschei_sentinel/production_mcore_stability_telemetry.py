from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class StabilityThresholds(StrictModel):
    schema_version: Literal["sentinel.mcore-stability-thresholds.v1"] = (
        "sentinel.mcore-stability-thresholds.v1"
    )
    max_grad_norm: float = Field(gt=0.0)
    max_cuda_allocated_fraction: float = Field(gt=0.0, le=1.0)
    min_expert_utilization_fraction: float = Field(ge=0.0, le=1.0)
    max_expert_load_cv: float = Field(ge=0.0)
    require_finite_parameters: bool = True
    require_finite_gradients: bool = True
    require_router_telemetry: bool = True


class MoERouterTelemetry(StrictModel):
    expert_count: int = Field(gt=0)
    tokens_per_expert: list[int] = Field(min_length=1)
    active_experts: int = Field(ge=0)
    utilization_fraction: float = Field(ge=0.0, le=1.0)
    load_mean: float = Field(ge=0.0)
    load_std: float = Field(ge=0.0)
    load_cv: float = Field(ge=0.0)
    min_tokens: int = Field(ge=0)
    max_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent(self) -> "MoERouterTelemetry":
        if len(self.tokens_per_expert) != self.expert_count:
            raise ValueError("tokens_per_expert length must equal expert_count")
        if self.active_experts != sum(1 for value in self.tokens_per_expert if value > 0):
            raise ValueError("active_experts mismatch")
        return self


class MCoreStabilityStepTelemetry(StrictModel):
    schema_version: Literal["sentinel.mcore-stability-step.v1"] = "sentinel.mcore-stability-step.v1"
    global_step: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    local_tokens: int = Field(gt=0)
    global_tokens: int = Field(gt=0)
    local_tokens_per_second: float = Field(gt=0.0)
    global_tokens_per_second: float = Field(gt=0.0)
    grad_norm: float = Field(ge=0.0)
    parameters_finite: bool
    gradients_finite: bool
    cuda_allocated_bytes: int = Field(ge=0)
    cuda_reserved_bytes: int = Field(ge=0)
    cuda_total_bytes: int = Field(gt=0)
    cuda_allocated_fraction: float = Field(ge=0.0, le=1.0)
    cuda_peak_allocated_bytes: int = Field(ge=0)
    aux_loss: float | None = None
    z_loss: float | None = None
    router: MoERouterTelemetry | None = None
    stability_passed: bool
    blockers: list[str] = Field(default_factory=list)
    execution_authorized: Literal[False] = False

    @model_validator(mode="after")
    def blockers_match_status(self) -> "MCoreStabilityStepTelemetry":
        if self.stability_passed != (len(self.blockers) == 0):
            raise ValueError("stability_passed must match blocker presence")
        return self


def summarize_router_tokens(tokens_per_expert: list[int]) -> MoERouterTelemetry:
    if not tokens_per_expert:
        raise ValueError("tokens_per_expert cannot be empty")
    if any(value < 0 for value in tokens_per_expert):
        raise ValueError("tokens_per_expert cannot contain negative counts")
    count = len(tokens_per_expert)
    active = sum(1 for value in tokens_per_expert if value > 0)
    mean = sum(tokens_per_expert) / count
    variance = sum((value - mean) ** 2 for value in tokens_per_expert) / count
    std = math.sqrt(variance)
    cv = std / mean if mean > 0 else 0.0
    return MoERouterTelemetry(
        expert_count=count,
        tokens_per_expert=tokens_per_expert,
        active_experts=active,
        utilization_fraction=active / count,
        load_mean=mean,
        load_std=std,
        load_cv=cv,
        min_tokens=min(tokens_per_expert),
        max_tokens=max(tokens_per_expert),
    )


def _finite_parameters_and_gradients(model: Any) -> tuple[bool, bool]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for stability telemetry") from exc
    parameters_finite = True
    gradients_finite = True
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() and not bool(torch.isfinite(parameter).all().item()):
                parameters_finite = False
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is not None and gradient.numel() and not bool(torch.isfinite(gradient).all().item()):
                gradients_finite = False
    return parameters_finite, gradients_finite


def collect_mcore_stability_telemetry(
    *,
    model: Any,
    global_step: int,
    local_tokens: int,
    global_tokens: int,
    elapsed_seconds: float,
    grad_norm: float,
    thresholds: StabilityThresholds,
    local_rank: int,
    aux_loss: float | None = None,
    z_loss: float | None = None,
    tokens_per_expert: list[int] | None = None,
) -> MCoreStabilityStepTelemetry:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for stability telemetry") from exc
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production stability telemetry")

    parameters_finite, gradients_finite = _finite_parameters_and_gradients(model)
    allocated = int(torch.cuda.memory_allocated(local_rank))
    reserved = int(torch.cuda.memory_reserved(local_rank))
    total = int(torch.cuda.get_device_properties(local_rank).total_memory)
    peak = int(torch.cuda.max_memory_allocated(local_rank))
    fraction = allocated / total
    router = summarize_router_tokens(tokens_per_expert) if tokens_per_expert is not None else None

    blockers: list[str] = []
    if thresholds.require_finite_parameters and not parameters_finite:
        blockers.append("non_finite_parameter")
    if thresholds.require_finite_gradients and not gradients_finite:
        blockers.append("non_finite_gradient")
    if not math.isfinite(grad_norm) or grad_norm > thresholds.max_grad_norm:
        blockers.append("grad_norm_threshold_exceeded")
    if fraction > thresholds.max_cuda_allocated_fraction:
        blockers.append("cuda_memory_threshold_exceeded")
    if aux_loss is not None and not math.isfinite(aux_loss):
        blockers.append("non_finite_moe_aux_loss")
    if z_loss is not None and not math.isfinite(z_loss):
        blockers.append("non_finite_moe_z_loss")
    if router is None:
        if thresholds.require_router_telemetry:
            blockers.append("router_telemetry_missing")
    else:
        if router.utilization_fraction < thresholds.min_expert_utilization_fraction:
            blockers.append("expert_utilization_below_threshold")
        if router.load_cv > thresholds.max_expert_load_cv:
            blockers.append("expert_load_imbalance")

    return MCoreStabilityStepTelemetry(
        global_step=global_step,
        elapsed_seconds=elapsed_seconds,
        local_tokens=local_tokens,
        global_tokens=global_tokens,
        local_tokens_per_second=local_tokens / elapsed_seconds,
        global_tokens_per_second=global_tokens / elapsed_seconds,
        grad_norm=max(0.0, grad_norm) if math.isfinite(grad_norm) else float("inf"),
        parameters_finite=parameters_finite,
        gradients_finite=gradients_finite,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
        cuda_total_bytes=total,
        cuda_allocated_fraction=fraction,
        cuda_peak_allocated_bytes=peak,
        aux_loss=aux_loss,
        z_loss=z_loss,
        router=router,
        stability_passed=not blockers,
        blockers=blockers,
    )
