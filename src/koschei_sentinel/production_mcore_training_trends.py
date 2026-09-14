from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from pydantic import Field, model_validator

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


class TrainingTrendState(StrictModel):
    schema_version: Literal["sentinel.mcore-training-trend-state.v1"] = (
        "sentinel.mcore-training-trend-state.v1"
    )
    history_window: int = Field(ge=3, le=1024)
    losses: list[float] = Field(default_factory=list)
    grad_norms: list[float] = Field(default_factory=list)
    activation_rms: list[float] = Field(default_factory=list)
    router_utilization: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self) -> "TrainingTrendState":
        for values in (self.losses, self.grad_norms, self.activation_rms, self.router_utilization):
            if len(values) > self.history_window:
                raise ValueError("trend history exceeds configured window")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("trend history must contain finite values")
        if any(value < 0.0 or value > 1.0 for value in self.router_utilization):
            raise ValueError("router utilization history outside [0,1]")
        return self


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

    @classmethod
    def from_state(
        cls,
        thresholds: TrainingTrendThresholds,
        state: TrainingTrendState,
    ) -> "TrainingTrendTracker":
        if state.history_window != thresholds.history_window:
            raise ValueError("trend history window does not match current thresholds")
        tracker = cls(thresholds)
        tracker._losses.extend(state.losses)
        tracker._grad_norms.extend(state.grad_norms)
        tracker._activation_rms.extend(state.activation_rms)
        tracker._router_utilization.extend(state.router_utilization)
        return tracker

    def to_state(self) -> TrainingTrendState:
        return TrainingTrendState(
            history_window=self.thresholds.history_window,
            losses=list(self._losses),
            grad_norms=list(self._grad_norms),
            activation_rms=list(self._activation_rms),
            router_utilization=list(self._router_utilization),
        )

    def preview(
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
            and router_base - worst_router_utilization_fraction > self.thresholds.max_router_utilization_drop
        ):
            blockers.append("router_utilization_collapse_trend")

        return TrainingTrendSnapshot(
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

    def commit_values(self, snapshot: TrainingTrendSnapshot, *, grad_norm: float) -> None:
        if not snapshot.trend_passed:
            raise ValueError("cannot commit a failed trend snapshot")
        if snapshot.lm_loss is not None and math.isfinite(snapshot.lm_loss):
            self._losses.append(float(snapshot.lm_loss))
        if math.isfinite(grad_norm):
            self._grad_norms.append(float(grad_norm))
        if snapshot.max_activation_rms is not None and math.isfinite(snapshot.max_activation_rms):
            self._activation_rms.append(float(snapshot.max_activation_rms))
        if snapshot.worst_router_utilization_fraction is not None:
            self._router_utilization.append(float(snapshot.worst_router_utilization_fraction))

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
        snapshot = self.preview(
            global_step=global_step,
            lm_loss=lm_loss,
            grad_norm=grad_norm,
            max_activation_rms=max_activation_rms,
            max_activation_abs=max_activation_abs,
            worst_router_utilization_fraction=worst_router_utilization_fraction,
        )
        if snapshot.trend_passed:
            self.commit_values(snapshot, grad_norm=grad_norm)
        return snapshot
