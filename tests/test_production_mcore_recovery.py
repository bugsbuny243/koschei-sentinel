from __future__ import annotations

import pytest

from koschei_sentinel.production_mcore_recovery import (
    RecoveryLedger,
    RecoveryPolicy,
    build_recovery_decision,
    fingerprint_batch,
)


def _policy() -> RecoveryPolicy:
    return RecoveryPolicy(
        max_retries_per_batch=2,
        learning_rate_backoff=0.5,
        min_learning_rate=1.0e-7,
        quarantine_after_failures=3,
        skip_quarantined_batch=True,
    )


def test_batch_fingerprint_is_deterministic_and_offset_sensitive():
    digest = "a" * 64
    first = fingerprint_batch(catalog_sha256=digest, global_sequence_offset=0, global_window=16)
    same = fingerprint_batch(catalog_sha256=digest, global_sequence_offset=0, global_window=16)
    shifted = fingerprint_batch(catalog_sha256=digest, global_sequence_offset=16, global_window=16)
    assert first == same
    assert first != shifted


def test_recovery_backoff_retries_then_quarantines():
    ledger = RecoveryLedger.empty()
    policy = _policy()
    fp = "b" * 64

    first = build_recovery_decision(
        globally_unstable=True,
        batch_fingerprint=fp,
        current_learning_rate=1.0e-5,
        policy=policy,
        ledger=ledger,
        reason_codes=["lm_loss_spike"],
    )
    assert first.failure_count == 1
    assert first.retry_allowed is True
    assert first.quarantine_batch is False
    assert first.next_learning_rate == pytest.approx(5.0e-6)
    assert ledger.total_retries == 1

    second = build_recovery_decision(
        globally_unstable=True,
        batch_fingerprint=fp,
        current_learning_rate=first.next_learning_rate,
        policy=policy,
        ledger=ledger,
        reason_codes=["lm_loss_spike"],
    )
    assert second.failure_count == 2
    assert second.retry_allowed is True
    assert second.next_learning_rate == pytest.approx(2.5e-6)
    assert ledger.total_retries == 2

    third = build_recovery_decision(
        globally_unstable=True,
        batch_fingerprint=fp,
        current_learning_rate=second.next_learning_rate,
        policy=policy,
        ledger=ledger,
        reason_codes=["lm_loss_spike"],
    )
    assert third.failure_count == 3
    assert third.retry_allowed is False
    assert third.quarantine_batch is True
    assert third.skip_batch is True
    assert fp in ledger.quarantined
    assert ledger.total_retries == 2


def test_recovery_learning_rate_stops_at_floor():
    ledger = RecoveryLedger.empty()
    decision = build_recovery_decision(
        globally_unstable=True,
        batch_fingerprint="c" * 64,
        current_learning_rate=1.2e-7,
        policy=_policy(),
        ledger=ledger,
        reason_codes=["grad_norm_spike"],
    )
    assert decision.next_learning_rate == pytest.approx(1.0e-7)
    assert ledger.total_retries == 1


def test_stable_consensus_does_not_mutate_ledger():
    ledger = RecoveryLedger.empty()
    decision = build_recovery_decision(
        globally_unstable=False,
        batch_fingerprint="d" * 64,
        current_learning_rate=1.0e-5,
        policy=_policy(),
        ledger=ledger,
        reason_codes=[],
    )
    assert decision.globally_unstable is False
    assert decision.offending_batch_fingerprint is None
    assert ledger.failures_by_fingerprint == {}
    assert ledger.quarantined == set()
    assert ledger.total_retries == 0


def test_invalid_quarantine_threshold_fails_closed():
    with pytest.raises(ValueError):
        RecoveryPolicy(
            max_retries_per_batch=1,
            learning_rate_backoff=0.5,
            min_learning_rate=1.0e-7,
            quarantine_after_failures=3,
            skip_quarantined_batch=True,
        )
