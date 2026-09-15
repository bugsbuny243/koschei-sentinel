from koschei_sentinel.production_mcore_recovery import RECOVERY_REASON_CODES, _canonical_recovery_reason


def test_recovery_reason_vocabulary_is_unique_and_stable():
    assert len(RECOVERY_REASON_CODES) == len(set(RECOVERY_REASON_CODES))
    assert "rank_exception" in RECOVERY_REASON_CODES
    assert "optimizer_step_failed" in RECOVERY_REASON_CODES
    assert "router_telemetry_missing" in RECOVERY_REASON_CODES


def test_rank_exception_details_collapse_to_fixed_code():
    assert _canonical_recovery_reason("rank_exception:OutOfMemoryError") == "rank_exception"
    assert _canonical_recovery_reason("rank_exception:RuntimeError") == "rank_exception"
    assert _canonical_recovery_reason("remote_rank_instability") == "rank_exception"


def test_unknown_reason_fails_closed_as_rank_exception():
    assert _canonical_recovery_reason("future_unknown_blocker") == "rank_exception"


def test_known_reason_is_preserved():
    assert _canonical_recovery_reason("lm_loss_spike") == "lm_loss_spike"
