from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.shadow_receipt import ShadowReplayReceipt
from koschei_sentinel.shadow_regression import (
    ShadowRegressionBlocked,
    ShadowRegressionTolerance,
    append_shadow_regression_history,
    build_shadow_regression_report,
    load_shadow_regression_history,
    write_shadow_regression_history,
)
from koschei_sentinel.shadow_review import ShadowReviewScorecard, ShadowReviewThresholds


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _receipt(candidate: str, *, replay: str = "a" * 64, results: str = "b" * 64) -> ShadowReplayReceipt:
    payload = {
        "schema_version": "sentinel.shadow-replay-receipt.v1",
        "candidate_id": candidate,
        "state": "completed_shadow_replay",
        "stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "plan_digest": "1" * 64,
        "proposal_digest": "2" * 64,
        "approval_digest": "3" * 64,
        "replay_sha256": replay,
        "replay_cases": 2,
        "results_path": f"build/shadow/{candidate}/results.jsonl",
        "results_sha256": results,
        "results_cases": 2,
        "output_dir": f"build/shadow/{candidate}",
        "complete_case_coverage": True,
        "ordered_case_identity_match": True,
        "manual_review_required": True,
        "benchmark_recheck_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReplayReceipt.model_validate({**payload, "receipt_digest": _digest(payload)})


def _scorecard(
    receipt: ShadowReplayReceipt,
    *,
    score: float = 1.0,
    failed: int = 0,
    followups: list[str] | None = None,
    gate: bool = True,
) -> ShadowReviewScorecard:
    followups = followups or []
    payload = {
        "schema_version": "sentinel.shadow-review-scorecard.v1",
        "candidate_id": receipt.candidate_id,
        "state": "reviewed_shadow_replay",
        "authority": "explanation_only",
        "receipt_digest": receipt.receipt_digest,
        "plan_digest": receipt.plan_digest,
        "results_sha256": receipt.results_sha256,
        "review_sha256": "c" * 64,
        "reviewers": ["reviewer@koschei"],
        "total_cases": 2,
        "passed_cases": 2 - failed,
        "failed_cases": failed,
        "case_pass_rate": score,
        "authority_score": score,
        "grounding_score": score,
        "abstention_score": score,
        "privacy_score": score,
        "followup_cases": followups,
        "thresholds": ShadowReviewThresholds().model_dump(mode="json"),
        "gate_passed": gate,
        "complete_manual_review": True,
        "benchmark_recheck_required": True,
        "owner_decision_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReviewScorecard.model_validate({**payload, "scorecard_digest": _digest(payload)})


def test_identical_or_better_candidate_passes_and_records_history(tmp_path: Path) -> None:
    baseline_receipt = _receipt("baseline", results="b" * 64)
    candidate_receipt = _receipt("candidate", results="d" * 64)
    report = build_shadow_regression_report(
        _scorecard(baseline_receipt),
        baseline_receipt,
        _scorecard(candidate_receipt),
        candidate_receipt,
    )
    assert report.regression_passed is True
    assert report.regression_reasons == []
    assert report.automatic_promotion_allowed is False
    assert report.production_deployment_allowed is False

    history = append_shadow_regression_history(report)
    path = tmp_path / "history.json"
    write_shadow_regression_history(history, path)
    loaded = load_shadow_regression_history(path)
    assert len(loaded.entries) == 1
    assert loaded.entries[0].report_digest == report.report_digest

    with pytest.raises(ShadowRegressionBlocked, match="already exists"):
        append_shadow_regression_history(report, loaded)


def test_score_drop_or_followup_growth_fails_closed() -> None:
    baseline_receipt = _receipt("baseline")
    candidate_receipt = _receipt("candidate", results="d" * 64)
    report = build_shadow_regression_report(
        _scorecard(baseline_receipt),
        baseline_receipt,
        _scorecard(candidate_receipt, score=0.5, failed=1, followups=["case-2"], gate=False),
        candidate_receipt,
    )
    assert report.regression_passed is False
    assert "candidate shadow review gate did not pass" in report.regression_reasons
    assert any("regressed beyond tolerance" in item for item in report.regression_reasons)
    assert "failed case count increased beyond tolerance" in report.regression_reasons
    assert "follow-up case count increased beyond tolerance" in report.regression_reasons


def test_explicit_tolerance_is_bounded_and_deterministic() -> None:
    baseline_receipt = _receipt("baseline")
    candidate_receipt = _receipt("candidate", results="d" * 64)
    report = build_shadow_regression_report(
        _scorecard(baseline_receipt),
        baseline_receipt,
        _scorecard(candidate_receipt, score=0.95),
        candidate_receipt,
        tolerance=ShadowRegressionTolerance(max_score_drop=0.05),
    )
    assert report.regression_passed is True
    assert report.case_pass_rate_delta == pytest.approx(-0.05)

    with pytest.raises(ValueError):
        ShadowRegressionTolerance(max_score_drop=-0.01)


def test_different_replay_or_thresholds_are_not_comparable() -> None:
    baseline_receipt = _receipt("baseline")
    candidate_receipt = _receipt("candidate", replay="e" * 64, results="d" * 64)
    with pytest.raises(ShadowRegressionBlocked, match="different sealed replay"):
        build_shadow_regression_report(
            _scorecard(baseline_receipt),
            baseline_receipt,
            _scorecard(candidate_receipt),
            candidate_receipt,
        )

    candidate_receipt = _receipt("candidate", results="d" * 64)
    candidate = _scorecard(candidate_receipt)
    payload = candidate.model_dump(mode="json")
    payload["thresholds"]["min_privacy_score"] = 0.5
    payload.pop("scorecard_digest")
    candidate = ShadowReviewScorecard.model_validate(
        {**payload, "scorecard_digest": _digest(payload)}
    )
    with pytest.raises(ShadowRegressionBlocked, match="different review thresholds"):
        build_shadow_regression_report(
            _scorecard(baseline_receipt),
            baseline_receipt,
            candidate,
            candidate_receipt,
        )


def test_tampered_scorecard_and_history_are_rejected(tmp_path: Path) -> None:
    baseline_receipt = _receipt("baseline")
    candidate_receipt = _receipt("candidate", results="d" * 64)
    candidate = _scorecard(candidate_receipt)
    payload = candidate.model_dump(mode="json")
    payload["privacy_score"] = 0.0
    tampered = ShadowReviewScorecard.model_construct(**payload)
    with pytest.raises(ShadowRegressionBlocked, match="scorecard digest"):
        build_shadow_regression_report(
            _scorecard(baseline_receipt),
            baseline_receipt,
            tampered,
            candidate_receipt,
        )

    report = build_shadow_regression_report(
        _scorecard(baseline_receipt), baseline_receipt, candidate, candidate_receipt
    )
    history = append_shadow_regression_history(report)
    path = tmp_path / "history.json"
    write_shadow_regression_history(history, path)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["entries"][0]["regression_passed"] = False
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ShadowRegressionBlocked, match="history digest"):
        load_shadow_regression_history(path)
