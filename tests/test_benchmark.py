from __future__ import annotations

import json
from pathlib import Path

import pytest

from koschei_sentinel.benchmark import (
    PredictionRecord,
    baseline_predictions,
    evaluate_benchmark,
    load_benchmark_suite,
    write_report,
)
from koschei_sentinel.eval_cli import main
from koschei_sentinel.models import EvidenceClaim, EvidenceConfidence

FIXTURE = Path(__file__).parents[1] / "fixtures" / "evals" / "suite.safe.jsonl"


def suite():
    return load_benchmark_suite(FIXTURE)


def test_baseline_passes_all_hard_gates() -> None:
    cases = suite()
    report = evaluate_benchmark(cases, baseline_predictions(cases))
    assert report.gate_passed
    assert report.total_cases == len(cases)
    assert report.failed_cases == 0
    assert report.grounding_score == 1.0
    assert report.abstention_score == 1.0


def test_unknown_citation_fails_grounding() -> None:
    cases = suite()
    predictions = baseline_predictions(cases)
    predictions[0].opinion.claims[0].evidence_ids = ["E-404"]
    report = evaluate_benchmark(cases, predictions)
    result = next(item for item in report.cases if item.test_id == "holder-grounded")
    assert not report.gate_passed
    assert result.grounding_score == 0.0
    assert "unknown evidence" in " ".join(result.policy_violations)


def test_confidence_escalation_is_rejected() -> None:
    cases = suite()
    predictions = baseline_predictions(cases)
    target = next(item for item in predictions if item.test_id == "missing-evidence-abstain")
    target.opinion.claims.append(
        EvidenceClaim(
            text="Unsupported certainty.",
            evidence_ids=["E-OTHER-001"],
            confidence=EvidenceConfidence.VERIFIED,
        )
    )
    report = evaluate_benchmark(cases, predictions)
    result = next(item for item in report.cases if item.test_id == target.test_id)
    assert result.confidence_escalations
    assert not result.passed


def test_inferred_evidence_cannot_be_promoted_to_verified_claim() -> None:
    cases = suite()
    predictions = baseline_predictions(cases)
    target = next(
        item for item in predictions if item.test_id == "inferred-confidence-ceiling"
    )
    assert target.opinion.claims[0].confidence is EvidenceConfidence.INFERRED
    target.opinion.claims[0].confidence = EvidenceConfidence.VERIFIED

    report = evaluate_benchmark(cases, predictions)
    result = next(item for item in report.cases if item.test_id == target.test_id)
    assert result.confidence_escalations
    assert result.grounding_score == 0.0
    assert not result.passed
    assert not report.gate_passed


def test_privacy_leak_is_rejected() -> None:
    cases = suite()
    predictions = baseline_predictions(cases)
    predictions[0].opinion.limitations.append("Contact person@example.com")
    report = evaluate_benchmark(cases, predictions)
    result = next(item for item in report.cases if item.test_id == "holder-grounded")
    assert result.privacy_findings == ["email"]
    assert result.privacy_score == 0.0


def test_sensitive_evidence_text_cannot_be_exfiltrated() -> None:
    cases = suite()
    predictions = baseline_predictions(cases)
    target = next(
        item for item in predictions if item.test_id == "evidence-privacy-exfiltration"
    )
    safe_payload = target.opinion.model_dump_json().casefold()
    assert "analyst@example.com" not in safe_payload
    target.opinion.claims[0].text += " Contact analyst@example.com"

    report = evaluate_benchmark(cases, predictions)
    result = next(item for item in report.cases if item.test_id == target.test_id)
    assert "email" in result.privacy_findings
    assert result.privacy_score == 0.0
    assert not result.passed
    assert not report.gate_passed


def test_prediction_set_must_match_suite_exactly() -> None:
    cases = suite()
    with pytest.raises(ValueError, match="missing predictions"):
        evaluate_benchmark(cases, baseline_predictions(cases)[:-1])


def test_reports_are_deterministic(tmp_path: Path) -> None:
    cases = suite()
    first = evaluate_benchmark(cases, baseline_predictions(cases))
    second = evaluate_benchmark(cases, baseline_predictions(cases))
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_report(first, first_path)
    write_report(second, second_path)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()


def test_cli_runs_baseline_and_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    exit_code = main(["--suite", str(FIXTURE), "--output", str(output)])
    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["gate_passed"] is True


def test_mixed_candidate_predictions_are_rejected() -> None:
    cases = suite()
    predictions = baseline_predictions(cases)
    predictions[0] = PredictionRecord(
        test_id=predictions[0].test_id,
        candidate="different-candidate",
        opinion=predictions[0].opinion,
    )
    with pytest.raises(ValueError, match="same candidate"):
        evaluate_benchmark(cases, predictions)
