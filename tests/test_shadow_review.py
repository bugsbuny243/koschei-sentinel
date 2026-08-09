from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koschei_sentinel.shadow_receipt import ShadowReplayReceipt
from koschei_sentinel.shadow_review import (
    ShadowReviewBlocked,
    ShadowReviewThresholds,
    build_shadow_review_scorecard,
    load_shadow_review_scorecard,
    write_shadow_review_scorecard,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _receipt(tmp_path: Path) -> ShadowReplayReceipt:
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    results = output / "results.jsonl"
    rows = [
        {"case_id": "case-1", "candidate_id": "sentinel-shadow-v1"},
        {"case_id": "case-2", "candidate_id": "sentinel-shadow-v1"},
    ]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "sentinel.shadow-replay-receipt.v1",
        "candidate_id": "sentinel-shadow-v1",
        "state": "completed_shadow_replay",
        "stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "plan_digest": "1" * 64,
        "proposal_digest": "2" * 64,
        "approval_digest": "3" * 64,
        "replay_sha256": "4" * 64,
        "replay_cases": 2,
        "results_path": "output/results.jsonl",
        "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
        "results_cases": 2,
        "output_dir": "output",
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


def _write_reviews(
    tmp_path: Path,
    receipt: ShadowReplayReceipt,
    *,
    second_grounding: bool = True,
    ids: tuple[str, str] = ("case-1", "case-2"),
) -> Path:
    review = tmp_path / "output" / "review.jsonl"
    rows = []
    for index, identifier in enumerate(ids):
        rows.append(
            {
                "schema_version": "sentinel.shadow-review.v1",
                "case_id": identifier,
                "candidate_id": receipt.candidate_id,
                "receipt_digest": receipt.receipt_digest,
                "reviewer_id": "reviewer@koschei",
                "authority_ok": True,
                "grounding_ok": second_grounding if index == 1 else True,
                "abstention_ok": True,
                "privacy_ok": True,
                "needs_followup": False,
                "notes": [],
            }
        )
    review.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return review


def test_complete_human_review_builds_deterministic_pass_scorecard(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    review = _write_reviews(tmp_path, receipt)

    first = build_shadow_review_scorecard(receipt, review_path=review, root=tmp_path)
    second = build_shadow_review_scorecard(receipt, review_path=review, root=tmp_path)

    assert first == second
    assert first.gate_passed is True
    assert first.case_pass_rate == 1.0
    assert first.grounding_score == 1.0
    assert first.complete_manual_review is True
    assert first.owner_decision_required is True
    assert first.automatic_promotion_allowed is False
    assert first.production_deployment_allowed is False


def test_manual_failure_is_aggregated_without_self_promotion(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    review = _write_reviews(tmp_path, receipt, second_grounding=False)

    scorecard = build_shadow_review_scorecard(
        receipt,
        review_path=review,
        root=tmp_path,
        thresholds=ShadowReviewThresholds(),
    )

    assert scorecard.gate_passed is False
    assert scorecard.passed_cases == 1
    assert scorecard.grounding_score == 0.5
    assert scorecard.automatic_promotion_allowed is False


def test_incomplete_or_reordered_review_fails_closed(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    review = _write_reviews(tmp_path, receipt, ids=("case-2", "case-1"))

    with pytest.raises(ShadowReviewBlocked, match="sealed order"):
        build_shadow_review_scorecard(receipt, review_path=review, root=tmp_path)


def test_wrong_receipt_binding_and_result_drift_fail_closed(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    review = _write_reviews(tmp_path, receipt)
    payload = json.loads(review.read_text(encoding="utf-8").splitlines()[0])
    payload["receipt_digest"] = "f" * 64
    lines = review.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(payload)
    review.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ShadowReviewBlocked, match="receipt_digest"):
        build_shadow_review_scorecard(receipt, review_path=review, root=tmp_path)

    review = _write_reviews(tmp_path, receipt)
    results = tmp_path / receipt.results_path
    results.write_text(results.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ShadowReviewBlocked, match="result bytes"):
        build_shadow_review_scorecard(receipt, review_path=review, root=tmp_path)


def test_scorecard_tamper_and_overwrite_are_rejected(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    review = _write_reviews(tmp_path, receipt)
    scorecard = build_shadow_review_scorecard(receipt, review_path=review, root=tmp_path)
    path = tmp_path / "scorecard.json"
    write_shadow_review_scorecard(scorecard, path)
    with pytest.raises(FileExistsError):
        write_shadow_review_scorecard(scorecard, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["automatic_promotion_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid shadow review scorecard"):
        load_shadow_review_scorecard(path)
