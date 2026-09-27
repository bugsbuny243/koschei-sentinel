from datetime import datetime, timezone

import pytest

from koschei_sentinel.defensive_recovery import DefensiveRecoveryCoordinator
from koschei_sentinel.defensive_response import DefensiveAction, DefensiveActionRequest


NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)


class Recovery:
    def __init__(self, verified=True):
        self.verified = verified
        self.calls = 0

    def rollback(self, request, execution):
        self.calls += 1
        return {"verified": self.verified}


def request():
    return DefensiveActionRequest(
        "a1", "t1", DefensiveAction.ISOLATE_WORKLOAD, "workload-1",
        ("f1",), ("e1",), "contain threat",
    )


def test_verified_effect_needs_no_rollback():
    recovery = Recovery()
    receipt = DefensiveRecoveryCoordinator(recovery_executor=recovery).finalize(
        request=request(), execution={"verified": True}, effect_verified=True, issued_at=NOW
    )
    assert receipt.effect_verified is True
    assert receipt.rolled_back is False
    assert recovery.calls == 0
    assert receipt.digest().startswith("sha256:")


def test_unverified_effect_rolls_back_and_records_receipt():
    recovery = Recovery()
    receipt = DefensiveRecoveryCoordinator(recovery_executor=recovery).finalize(
        request=request(), execution={"verified": True}, effect_verified=False, issued_at=NOW
    )
    assert receipt.rolled_back is True
    assert receipt.rollback_verified is True
    assert recovery.calls == 1


def test_unverified_rollback_fails_closed():
    recovery = Recovery(verified=False)
    with pytest.raises(RuntimeError, match="rollback could not be verified"):
        DefensiveRecoveryCoordinator(recovery_executor=recovery).finalize(
            request=request(), execution={"verified": True}, effect_verified=False, issued_at=NOW
        )


def test_unverified_execution_cannot_be_finalized():
    with pytest.raises(RuntimeError, match="unverified defensive execution"):
        DefensiveRecoveryCoordinator(recovery_executor=Recovery()).finalize(
            request=request(), execution={"verified": False}, effect_verified=True, issued_at=NOW
        )
