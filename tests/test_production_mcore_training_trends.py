from __future__ import annotations

from koschei_sentinel.production_mcore_training_trends import (
    TrainingTrendThresholds,
    TrainingTrendTracker,
)


def _tracker() -> TrainingTrendTracker:
    return TrainingTrendTracker(
        TrainingTrendThresholds(
            history_window=8,
            min_history_for_spike=3,
            max_loss_ratio_to_median=2.0,
            max_grad_norm_ratio_to_median=3.0,
            max_activation_rms_ratio_to_median=3.0,
            max_activation_abs=100.0,
            max_router_utilization_drop=0.25,
        )
    )


def _observe(tracker, step, *, loss=2.0, grad=1.0, rms=2.0, max_abs=4.0, util=0.9):
    return tracker.observe(
        global_step=step,
        lm_loss=loss,
        grad_norm=grad,
        max_activation_rms=rms,
        max_activation_abs=max_abs,
        worst_router_utilization_fraction=util,
    )


def test_baseline_warmup_does_not_flag_normal_steps():
    tracker = _tracker()
    for step in range(1, 4):
        snapshot = _observe(tracker, step)
        assert snapshot.trend_passed is True
        assert snapshot.blockers == []


def test_loss_spike_is_detected_against_history_median():
    tracker = _tracker()
    for step in range(1, 4):
        _observe(tracker, step)
    snapshot = _observe(tracker, 4, loss=5.0)
    assert "lm_loss_spike" in snapshot.blockers


def test_grad_and_activation_spikes_are_detected():
    tracker = _tracker()
    for step in range(1, 4):
        _observe(tracker, step)
    snapshot = _observe(tracker, 4, grad=4.0, rms=7.0)
    assert "grad_norm_spike" in snapshot.blockers
    assert "activation_rms_spike" in snapshot.blockers


def test_absolute_activation_threshold_is_detected_without_history():
    tracker = _tracker()
    snapshot = _observe(tracker, 1, max_abs=101.0)
    assert "activation_abs_threshold_exceeded" in snapshot.blockers


def test_router_collapse_trend_is_detected():
    tracker = _tracker()
    for step in range(1, 4):
        _observe(tracker, step, util=0.9)
    snapshot = _observe(tracker, 4, util=0.5)
    assert "router_utilization_collapse_trend" in snapshot.blockers


def test_failed_preview_does_not_mutate_history():
    tracker = _tracker()
    for step in range(1, 4):
        _observe(tracker, step)
    before = tracker.to_state()
    snapshot = tracker.preview(
        global_step=4,
        lm_loss=20.0,
        grad_norm=10.0,
        max_activation_rms=20.0,
        max_activation_abs=200.0,
        worst_router_utilization_fraction=0.1,
    )
    assert snapshot.trend_passed is False
    assert tracker.to_state() == before


def test_preview_commits_only_after_explicit_consensus_commit():
    tracker = _tracker()
    snapshot = tracker.preview(
        global_step=1,
        lm_loss=2.0,
        grad_norm=1.0,
        max_activation_rms=2.0,
        max_activation_abs=4.0,
        worst_router_utilization_fraction=0.9,
    )
    assert tracker.to_state().losses == []
    tracker.commit_values(snapshot, grad_norm=1.0)
    assert tracker.to_state().losses == [2.0]
