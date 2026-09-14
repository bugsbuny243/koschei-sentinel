from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel


class TrainingTrendThresholds(StrictModel):
    schema_version: Literal["sentinel.mcore-training-trend-thresholds.v1"] = (
        "sentinel.mcore-training-trend-thresholds.v1"
    )
    history_window: int = Field(ge=3, le=1024)
    min_history_for_spike: int = Field(ge=2)
    max_loss_ratio_to_median: float = Field(gt=1.0)
    max_grad_norm_ratio_to_median: float = Field(gt=1.0)
    max_activation_rms_ratio_to_median: float = Field(gt=1.0)
    max_activation_abs: float = Field(gt=0.0)
    max_router_utilization_drop: float = Field(gt=0.0, le=1.0)


class TrainingTrendSnapshot(StrictModel):
    schema_version: Literal["sentinel.mcore-training-trend.v1"] = "sentinel.mcore-training-trend.v1"
    global_step: int = Field(gt=0)
    lm_loss: float | None = Field(default=None, ge=0.0)
    loss_baseline_median: float | None = Field(default=None, ge=0.0)
    loss_ratio_to_median: float | None = Field(default=None, ge=0.0)
    grad_norm_baseline_median: float | None = Field(default=None, ge=0.0)
    grad_norm_ratio_to_median: float | None = Field(default=None, ge=0.0)
    max_activation_rms: float | None = Field(default=None, ge=0.0)
    activation_rms_baseline_median: float | None = Field(default=None, ge=0.0)
    activation_rms_ratio_to_median: float | None = Field(default=None, ge=0.0)
    max_activation_abs: float | None = Field(default=None, ge=0.0)
    worst_router_utilization_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    router_utilization_baseline_median: float | None = Field(default=None, ge=0.0, le=1.0)
    blockers: list[str] = Field(default_factory=list)
    trend_passed: bool
    execution_authorized: Literal[False] = False


def _finite(value: float | None) -> bool:
    return value is None or math.isfinite(value)


def _baseline(values: deque[float], minimum: int) -> float | None:
    if len(values) < minimum:
        return None
    return float(median(values))


@dataclass
class TrainingTrendTracker:
    thresholds: TrainingTrendThresholds
    _losses: deque[float] = field(init=False)
    _grad_norms: deque[float] = field(init=False)
    _activation_rms: deque[float] = field(init=False)
    _router_utilization: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        size = self.thresholds.history_window
        self._losses = deque(maxlen=size)
        self._grad_norms = deque(maxlen=size)
        self._activation_rms = deque(maxlen=size)
        self._router_utilization = deque(maxlen=size)

    def observe(
        self,
        *,
        global_step: int,
        lm_loss: float | None,
        grad_norm: float,
        max_activation_rms: float | None,
        max_activation_abs: float | None,
        worst_router_utilization_fraction: float | None,
    ) -> TrainingTrendSnapshot:
        minimum = self.thresholds.min_history_for_spike
        loss_base = _baseline(self._losses, minimum)
        grad_base = _baseline(self._grad_norms, minimum)
        act_base = _baseline(self._activation_rms, minimum)
        router_base = _baseline(self._router_utilization, minimum)

        loss_ratio = lm_loss / loss_base if lm_loss is not None and loss_base and loss_base > 0 else None
        grad_ratio = grad_norm / grad_base if grad_base and grad_base > 0 else None
        act_ratio = (
            max_activation_rms / act_base
            if max_activation_rms is not None and act_base and act_base > 0
            else None
        )

        blockers: list[str] = []
        if not _finite(lm_loss):
            blockers.append("non_finite_lm_loss")
        if not math.isfinite(grad_norm):
            blockers.append("non_finite_grad_norm")
        if not _finite(max_activation_rms) or not _finite(max_activation_abs):
            blockers.append("non_finite_activation")
        if loss_ratio is not None and loss_ratio > self.thresholds.max_loss_ratio_to_median:
            blockers.append("lm_loss_spike")
        if grad_ratio is not None and grad_ratio > self.thresholds.max_grad_norm_ratio_to_median:
            blockers.append("grad_norm_spike")
        if act_ratio is not None and act_ratio > self.thresholds.max_activation_rms_ratio_to_median:
            blockers.append("activation_rms_spike")
        if max_activation_abs is not None and max_activation_abs > self.thresholds.max_activation_abs:
            blockers.append("activation_abs_threshold_exceeded")
        if (
            worst_router_utilization_fraction is not None
            and router_base is not None
            and router_base - worst_router_utilization_fraction
            > self.thresholds.max_router_utilization_drop
        ):
            blockers.append("router_utilization_collapse_trend")

        snapshot = TrainingTrendSnapshot(
            global_step=global_step,
            lm_loss=lm_loss,
            loss_baseline_median=loss_base,
            loss_ratio_to_median=loss_ratio,
            grad_norm_baseline_median=grad_base,
            grad_norm_ratio_to_median=grad_ratio,
            max_activation_rms=max_activation_rms,
            activation_rms_baseline_median=act_base,
            activation_rms_ratio_to_median=act_ratio,
            max_activation_abs=max_activation_abs,
            worst_router_utilization_fraction=worst_router_utilization_fraction,
            router_utilization_baseline_median=router_base,
            blockers=blockers,
            trend_passed=not blockers,
        )

        if lm_loss is not None and math.isfinite(lm_loss):
            self._losses.append(float(lm_loss))
        if math.isfinite(grad_norm):
            self._grad_norms.append(float(grad_norm))
        if max_activation_rms is not None and math.isfinite(max_activation_rms):
            self._activation_rms.append(float(max_activation_rms))
        if worst_router_utilization_fraction is not None:
            self._router_utilization.append(float(worst_router_utilization_fraction))
        return snapshot
