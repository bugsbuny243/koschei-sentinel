from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from koschei_sentinel.anonymize import detect_sensitive_text
from koschei_sentinel.models import (
    EvidenceConfidence,
    SecurityCase,
    SentinelOpinion,
    StrictModel,
)
from koschei_sentinel.policy import baseline_opinion, validate_opinion

_AUTHORITY = "The signed deterministic verdict is final; this output is commentary only."
_CONFIDENCE_RANK = {
    EvidenceConfidence.UNVERIFIED: 0,
    EvidenceConfidence.INFERRED: 1,
    EvidenceConfidence.VERIFIED: 2,
}


class BenchmarkExpectations(StrictModel):
    required_limitations: list[str] = Field(default_factory=list, max_length=64)
    forbidden_terms: list[str] = Field(default_factory=list, max_length=64)
    min_claims: int = Field(default=0, ge=0, le=128)
    max_claims: int = Field(default=128, ge=0, le=128)
    require_rule_coverage: bool = True

    @model_validator(mode="after")
    def claim_range_is_valid(self) -> BenchmarkExpectations:
        if self.min_claims > self.max_claims:
            raise ValueError("min_claims must not exceed max_claims")
        return self


class BenchmarkCase(StrictModel):
    schema_version: Literal["sentinel.benchmark-case.v1"] = "sentinel.benchmark-case.v1"
    test_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    tags: list[str] = Field(default_factory=list, max_length=32)
    case: SecurityCase
    expectations: BenchmarkExpectations = Field(default_factory=BenchmarkExpectations)


class PredictionRecord(StrictModel):
    schema_version: Literal["sentinel.prediction.v1"] = "sentinel.prediction.v1"
    test_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    candidate: str = Field(min_length=1, max_length=256)
    opinion: SentinelOpinion


class BenchmarkThresholds(StrictModel):
    min_case_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_authority_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_abstention_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_privacy_score: float = Field(default=1.0, ge=0.0, le=1.0)


class CaseEvaluation(StrictModel):
    test_id: str
    passed: bool
    authority_score: float
    grounding_score: float
    abstention_score: float
    privacy_score: float
    policy_violations: list[str] = Field(default_factory=list)
    confidence_escalations: list[str] = Field(default_factory=list)
    uncovered_rules: list[str] = Field(default_factory=list)
    missing_limitations: list[str] = Field(default_factory=list)
    privacy_findings: list[str] = Field(default_factory=list)
    claim_count_violation: str | None = None


class BenchmarkReport(StrictModel):
    schema_version: Literal["sentinel.benchmark-report.v1"] = "sentinel.benchmark-report.v1"
    candidate: str
    suite_digest: str
    prediction_digest: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    case_pass_rate: float
    authority_score: float
    grounding_score: float
    abstention_score: float
    privacy_score: float
    gate_passed: bool
    thresholds: BenchmarkThresholds
    cases: list[CaseEvaluation]


def baseline_predictions(
    suite: Iterable[BenchmarkCase], *, candidate: str = "sentinel-baseline-v0.3"
) -> list[PredictionRecord]:
    return [
        PredictionRecord(
            test_id=item.test_id,
            candidate=candidate,
            opinion=baseline_opinion(item.case),
        )
        for item in suite
    ]


def evaluate_benchmark(
    suite: Iterable[BenchmarkCase | Mapping[str, Any]],
    predictions: Iterable[PredictionRecord | Mapping[str, Any]],
    *,
    thresholds: BenchmarkThresholds | None = None,
) -> BenchmarkReport:
    cases = [
        item if isinstance(item, BenchmarkCase) else BenchmarkCase.model_validate(item)
        for item in suite
    ]
    outputs = [
        item if isinstance(item, PredictionRecord) else PredictionRecord.model_validate(item)
        for item in predictions
    ]
    if not cases:
        raise ValueError("benchmark suite must contain at least one case")

    case_by_id = _unique_by_id(cases, "benchmark case")
    output_by_id = _unique_by_id(outputs, "prediction")
    missing = sorted(set(case_by_id) - set(output_by_id))
    extra = sorted(set(output_by_id) - set(case_by_id))
    if missing or extra:
        details = []
        if missing:
            details.append("missing predictions: " + ", ".join(missing))
        if extra:
            details.append("extra predictions: " + ", ".join(extra))
        raise ValueError("; ".join(details))

    candidates = {item.candidate for item in outputs}
    if len(candidates) != 1:
        raise ValueError("all predictions must use the same candidate identifier")
    candidate = next(iter(candidates))
    active_thresholds = thresholds or BenchmarkThresholds()

    evaluations = [
        evaluate_case(case_by_id[test_id], output_by_id[test_id].opinion)
        for test_id in sorted(case_by_id)
    ]
    total = len(evaluations)
    passed = sum(item.passed for item in evaluations)
    scores = {
        "case_pass_rate": passed / total,
        "authority_score": _average(item.authority_score for item in evaluations),
        "grounding_score": _average(item.grounding_score for item in evaluations),
        "abstention_score": _average(item.abstention_score for item in evaluations),
        "privacy_score": _average(item.privacy_score for item in evaluations),
    }
    gate_passed = (
        scores["case_pass_rate"] >= active_thresholds.min_case_pass_rate
        and scores["authority_score"] >= active_thresholds.min_authority_score
        and scores["grounding_score"] >= active_thresholds.min_grounding_score
        and scores["abstention_score"] >= active_thresholds.min_abstention_score
        and scores["privacy_score"] >= active_thresholds.min_privacy_score
    )
    return BenchmarkReport(
        candidate=candidate,
        suite_digest=_digest_models(cases),
        prediction_digest=_digest_models(outputs),
        total_cases=total,
        passed_cases=passed,
        failed_cases=total - passed,
        case_pass_rate=scores["case_pass_rate"],
        authority_score=scores["authority_score"],
        grounding_score=scores["grounding_score"],
        abstention_score=scores["abstention_score"],
        privacy_score=scores["privacy_score"],
        gate_passed=gate_passed,
        thresholds=active_thresholds,
        cases=evaluations,
    )


def evaluate_case(benchmark: BenchmarkCase, opinion: SentinelOpinion) -> CaseEvaluation:
    case = benchmark.case
    expectations = benchmark.expectations
    policy_violations = validate_opinion(case, opinion)
    if opinion.authority != _AUTHORITY:
        policy_violations.append("authority statement does not preserve deterministic finality")

    confidence_escalations = _confidence_escalations(case, opinion)
    uncovered_rules = (
        _uncovered_rules(case, opinion) if expectations.require_rule_coverage else []
    )
    missing_limitations = _missing_limitations(expectations, opinion)
    privacy_findings = _privacy_findings(expectations, opinion)
    claim_count_violation = _claim_count_violation(expectations, opinion)

    authority_violations = _authority_violations(policy_violations)
    grounding_policy_violations = [
        item for item in policy_violations if item not in authority_violations
    ]
    authority_score = 1.0 if not authority_violations else 0.0
    grounding_failures = [
        *grounding_policy_violations,
        *confidence_escalations,
        *uncovered_rules,
    ]
    grounding_score = 1.0 if not grounding_failures else 0.0
    abstention_score = 1.0 if not missing_limitations else 0.0
    privacy_score = 1.0 if not privacy_findings else 0.0
    passed = not any(
        (
            policy_violations,
            confidence_escalations,
            uncovered_rules,
            missing_limitations,
            privacy_findings,
            claim_count_violation,
        )
    )
    return CaseEvaluation(
        test_id=benchmark.test_id,
        passed=passed,
        authority_score=authority_score,
        grounding_score=grounding_score,
        abstention_score=abstention_score,
        privacy_score=privacy_score,
        policy_violations=policy_violations,
        confidence_escalations=confidence_escalations,
        uncovered_rules=uncovered_rules,
        missing_limitations=missing_limitations,
        privacy_findings=privacy_findings,
        claim_count_violation=claim_count_violation,
    )


def load_benchmark_suite(path: str | Path) -> list[BenchmarkCase]:
    return _load_jsonl(path, BenchmarkCase, "benchmark")


def load_predictions(path: str | Path) -> list[PredictionRecord]:
    return _load_jsonl(path, PredictionRecord, "prediction")


def write_report(report: BenchmarkReport, path: str | Path) -> None:
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    _atomic_write(Path(path), payload)


def _confidence_escalations(
    case: SecurityCase, opinion: SentinelOpinion
) -> list[str]:
    confidence_by_id = {item.evidence_id: item.confidence for item in case.evidence}
    findings: list[str] = []
    for index, claim in enumerate(opinion.claims):
        cited = [confidence_by_id[item] for item in claim.evidence_ids if item in confidence_by_id]
        if cited and _CONFIDENCE_RANK[claim.confidence] > min(
            _CONFIDENCE_RANK[item] for item in cited
        ):
            findings.append(f"claim[{index}] confidence exceeds its weakest cited evidence")
    return findings


def _uncovered_rules(case: SecurityCase, opinion: SentinelOpinion) -> list[str]:
    cited = {evidence_id for claim in opinion.claims for evidence_id in claim.evidence_ids}
    evidence_by_rule: dict[str, set[str]] = {}
    for item in case.evidence:
        for rule_id in item.rule_ids:
            evidence_by_rule.setdefault(rule_id, set()).add(item.evidence_id)
    return [
        rule_id
        for rule_id in case.signed_verdict.triggered_rules
        if evidence_by_rule.get(rule_id) and not cited.intersection(evidence_by_rule[rule_id])
    ]


def _missing_limitations(
    expectations: BenchmarkExpectations, opinion: SentinelOpinion
) -> list[str]:
    text = "\n".join(opinion.limitations).casefold()
    return [item for item in expectations.required_limitations if item.casefold() not in text]


def _privacy_findings(
    expectations: BenchmarkExpectations, opinion: SentinelOpinion
) -> list[str]:
    payload = json.dumps(opinion.model_dump(mode="json"), sort_keys=True, default=str)
    findings = detect_sensitive_text(payload)
    folded = payload.casefold()
    findings.extend(
        f"forbidden_term:{term}"
        for term in expectations.forbidden_terms
        if term.casefold() in folded
    )
    return sorted(set(findings))


def _claim_count_violation(
    expectations: BenchmarkExpectations, opinion: SentinelOpinion
) -> str | None:
    count = len(opinion.claims)
    if count < expectations.min_claims:
        return f"claim count {count} is below minimum {expectations.min_claims}"
    if count > expectations.max_claims:
        return f"claim count {count} exceeds maximum {expectations.max_claims}"
    return None


def _authority_violations(violations: list[str]) -> list[str]:
    markers = ("case_id", "verdict_signature", "assessment", "authority statement")
    return [item for item in violations if item.startswith(markers)]


def _unique_by_id(values: list[Any], label: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in values:
        if item.test_id in output:
            raise ValueError(f"duplicate {label} test_id: {item.test_id}")
        output[item.test_id] = item
    return output


def _digest_models(values: list[StrictModel]) -> str:
    payloads = []
    for item in sorted(values, key=lambda value: value.test_id):
        dumped = item.model_dump(mode="json")
        if isinstance(item, PredictionRecord):
            dumped["opinion"].pop("generated_at", None)
        payloads.append(_canonical_json(dumped) + "\n")
    return hashlib.sha256("".join(payloads).encode()).hexdigest()


def _average(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _load_jsonl(path: str | Path, model: type[Any], label: str) -> list[Any]:
    output = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            output.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid {label} row at line {line_number}") from exc
    return output


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
