from __future__ import annotations

import pytest

from koschei_sentinel.production_foundation_catalog import FoundationCatalogCursor
from koschei_sentinel.production_mcore_recovery import RecoveryLedger
from koschei_sentinel.production_mcore_resume_bundle import (
    attach_catalog_resume_state,
    restore_catalog_resume_bundle,
)
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState
from koschei_sentinel.production_mcore_training_trends import (
    TrainingTrendThresholds,
    TrainingTrendTracker,
)


CATALOG = "a" * 64
OTHER = "b" * 64


def thresholds(window: int = 8) -> TrainingTrendThresholds:
    return TrainingTrendThresholds(
        history_window=window,
        min_history_for_spike=2,
        max_loss_ratio_to_median=2.5,
        max_grad_norm_ratio_to_median=4.0,
        max_activation_rms_ratio_to_median=4.0,
        max_activation_abs=1.0e4,
        max_router_utilization_drop=0.35,
    )


def base_state() -> MCoreTrainingState:
    return MCoreTrainingState(
        global_step=3,
        consumed_microbatches=6,
        learning_rate=5.0e-6,
        global_seed=39735,
        data_state=FoundationCatalogCursor(
            global_sequence_offset=96,
            catalog_sha256=CATALOG,
            data_parallel_size=16,
        ).model_dump(mode="json"),
    )


def test_v2_resume_restores_ledger_trend_and_quarantine_history():
    ledger = RecoveryLedger.empty()
    fingerprint = "c" * 64
    ledger.record_failure(fingerprint)
    ledger.record_retry()
    tracker = TrainingTrendTracker(thresholds())
    snapshot = tracker.preview(
        global_step=1,
        lm_loss=2.0,
        grad_norm=1.0,
        max_activation_rms=0.5,
        max_activation_abs=1.0,
        worst_router_utilization_fraction=0.9,
    )
    tracker.commit_values(snapshot, grad_norm=1.0)
    quarantine_event = {
        "batch_fingerprint": fingerprint,
        "global_sequence_offset": 64,
        "global_window": 32,
        "failure_count": 3,
        "skipped": True,
        "reason_codes": ["lm_loss_spike"],
    }
    state = attach_catalog_resume_state(
        base_state(),
        catalog_sha256=CATALOG,
        current_learning_rate=5.0e-6,
        ledger=ledger,
        trend_tracker=tracker,
        quarantine_events=[quarantine_event],
    )

    restored = restore_catalog_resume_bundle(
        state,
        expected_catalog_sha256=CATALOG,
        trend_thresholds=thresholds(),
    )
    assert restored.cursor is not None
    assert restored.cursor.global_sequence_offset == 96
    assert restored.ledger.failures_by_fingerprint[fingerprint] == 1
    assert restored.ledger.total_retries == 1
    assert restored.total_retries == 1
    assert restored.quarantine_events == [quarantine_event]
    assert restored.trend_tracker.to_state().losses == [2.0]
    assert restored.current_learning_rate == 5.0e-6


def test_v1_style_state_falls_back_to_empty_recovery_and_trend():
    state = base_state().model_copy(
        update={
            "schema_version": "sentinel.mcore-training-state.v1",
            "recovery_state": None,
            "trend_state": None,
        }
    )
    restored = restore_catalog_resume_bundle(
        state,
        expected_catalog_sha256=CATALOG,
        trend_thresholds=thresholds(),
    )
    assert restored.ledger.failures_by_fingerprint == {}
    assert restored.ledger.quarantined == set()
    assert restored.total_retries == 0
    assert restored.quarantine_events == []
    assert restored.trend_tracker.to_state().losses == []


def test_resume_rejects_catalog_digest_mismatch():
    with pytest.raises(ValueError, match="catalog cursor digest mismatch"):
        restore_catalog_resume_bundle(
            base_state(),
            expected_catalog_sha256=OTHER,
            trend_thresholds=thresholds(),
        )


def test_resume_rejects_learning_rate_disagreement():
    ledger = RecoveryLedger.empty()
    tracker = TrainingTrendTracker(thresholds())
    state = attach_catalog_resume_state(
        base_state(),
        catalog_sha256=CATALOG,
        current_learning_rate=5.0e-6,
        ledger=ledger,
        trend_tracker=tracker,
    )
    state = state.model_copy(update={"learning_rate": 1.0e-5})
    with pytest.raises(ValueError, match="learning-rate state mismatch"):
        restore_catalog_resume_bundle(
            state,
            expected_catalog_sha256=CATALOG,
            trend_thresholds=thresholds(),
        )


def test_resume_rejects_changed_trend_window():
    tracker = TrainingTrendTracker(thresholds(window=8))
    state = attach_catalog_resume_state(
        base_state(),
        catalog_sha256=CATALOG,
        current_learning_rate=5.0e-6,
        ledger=RecoveryLedger.empty(),
        trend_tracker=tracker,
    )
    with pytest.raises(ValueError, match="trend history window"):
        restore_catalog_resume_bundle(
            state,
            expected_catalog_sha256=CATALOG,
            trend_thresholds=thresholds(window=16),
        )


def test_resume_rejects_retry_counter_disagreement():
    ledger = RecoveryLedger.empty()
    ledger.record_retry()
    tracker = TrainingTrendTracker(thresholds())
    state = attach_catalog_resume_state(
        base_state(),
        catalog_sha256=CATALOG,
        current_learning_rate=5.0e-6,
        ledger=ledger,
        trend_tracker=tracker,
    )
    raw = dict(state.recovery_state or {})
    raw["total_retries"] = 0
    state = state.model_copy(update={"recovery_state": raw})
    with pytest.raises(ValueError, match="retry count disagrees"):
        restore_catalog_resume_bundle(
            state,
            expected_catalog_sha256=CATALOG,
            trend_thresholds=thresholds(),
        )
