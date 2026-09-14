from __future__ import annotations

import pytest

from koschei_sentinel.production_mcore_stability_telemetry import (
    MCoreStabilityStepTelemetry,
    StabilityThresholds,
    summarize_router_tokens,
)


def test_router_summary_balanced_load():
    value = summarize_router_tokens([10, 10, 10, 10])
    assert value.active_experts == 4
    assert value.utilization_fraction == 1.0
    assert value.load_cv == 0.0


def test_router_summary_detects_partial_utilization():
    value = summarize_router_tokens([20, 0, 0, 20])
    assert value.active_experts == 2
    assert value.utilization_fraction == 0.5
    assert value.load_cv > 0.0


def test_step_telemetry_rejects_status_without_blocker_coherence():
    with pytest.raises(ValueError):
        MCoreStabilityStepTelemetry(
            global_step=1,
            elapsed_seconds=1.0,
            local_tokens=16,
            global_tokens=32,
            local_tokens_per_second=16.0,
            global_tokens_per_second=32.0,
            grad_norm=1.0,
            parameters_finite=True,
            gradients_finite=True,
            cuda_allocated_bytes=1,
            cuda_reserved_bytes=1,
            cuda_total_bytes=2,
            cuda_allocated_fraction=0.5,
            cuda_peak_allocated_bytes=1,
            stability_passed=True,
            blockers=["should_fail"],
        )


def test_threshold_contract_is_strict():
    thresholds = StabilityThresholds(
        max_grad_norm=10.0,
        max_cuda_allocated_fraction=0.9,
        min_expert_utilization_fraction=0.75,
        max_expert_load_cv=1.0,
    )
    assert thresholds.require_finite_parameters is True
    assert thresholds.require_finite_gradients is True
