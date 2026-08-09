from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from koschei_sentinel.models import StrictModel
from koschei_sentinel.shadow_receipt import ShadowReplayReceipt
from koschei_sentinel.shadow_review import ShadowReviewScorecard, ShadowReviewThresholds

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_MAX_HISTORY = 10_000


class ShadowRegressionBlocked(ValueError):
    """Raised when two shadow scorecards are not safely comparable."""


class ShadowRegressionTolerance(StrictModel):
    max_score_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    max_failed_case_increase: int = Field(default=0, ge=0)
    max_followup_case_increase: int = Field(default=0, ge=0)


class ShadowRegressionReport(StrictModel):
    schema_version: Literal["sentinel.shadow-regression-report.v1"] = (
        "sentinel.shadow-regression-report.v1"
    )
    baseline_candidate_id: str = Field(pattern=_CANDIDATE_ID)
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    baseline_scorecard_digest: str = Field(pattern=_DIGEST)
    candidate_scorecard_digest: str = Field(pattern=_DIGEST)
    baseline_receipt_digest: str = Field(pattern=_DIGEST)
    candidate_receipt_digest: str = Field(pattern=_DIGEST)
    replay_sha256: str = Field(pattern=_DIGEST)
    replay_cases: int = Field(ge=1)
    thresholds: ShadowReviewThresholds
    tolerance: ShadowRegressionTolerance
    case_pass_rate_delta: float = Field(ge=-1.0, le=1.0)
    authority_score_delta: float = Field(ge=-1.0, le=1.0)
    grounding_score_delta: float = Field(ge=-1.0, le=1.0)
    abstention_score_delta: float = Field(ge=-1.0, le=1.0)
    privacy_score_delta: float = Field(ge=-1.0, le=1.0)
    failed_case_delta: int
    followup_case_delta: int
    regression_reasons: list[str] = Field(default_factory=list, max_length=32)
    regression_passed: bool
    manual_review_required: Literal[True] = True
    benchmark_recheck_required: Literal[True] = True
    owner_decision_required: Literal[True] = True
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    verdict_mutation_allowed: Literal[False] = False
    report_digest: str = Field(pattern=_DIGEST)


class ShadowRegressionHistoryEntry(StrictModel):
    baseline_candidate_id: str = Field(pattern=_CANDIDATE_ID)
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    replay_sha256: str = Field(pattern=_DIGEST)
    report_digest: str = Field(pattern=_DIGEST)
    regression_passed: bool


class ShadowRegressionHistory(StrictModel):
    schema_version: Literal["sentinel.shadow-regression-history.v1"] = (
        "sentinel.shadow-regression-history.v1"
    )
    entries: list[ShadowRegressionHistoryEntry] = Field(
        default_factory=list,
        max_length=_MAX_HISTORY,
    )
    automatic_promotion_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    history_digest: str = Field(pattern=_DIGEST)


def build_shadow_regression_report(
    baseline: ShadowReviewScorecard,
    baseline_receipt: ShadowReplayReceipt,
    candidate: ShadowReviewScorecard,
    candidate_receipt: ShadowReplayReceipt,
    *,
    tolerance: ShadowRegressionTolerance | None = None,
) -> ShadowRegressionReport:
    _require_scorecard_digest(baseline)
    _require_scorecard_digest(candidate)
    _require_receipt_digest(baseline_receipt)
    _require_receipt_digest(candidate_receipt)
    _require_scorecard_receipt_binding(baseline, baseline_receipt, "baseline")
    _require_scorecard_receipt_binding(candidate, candidate_receipt, "candidate")

    if baseline_receipt.replay_sha256 != candidate_receipt.replay_sha256:
        raise ShadowRegressionBlocked("shadow scorecards use different sealed replay bytes")
    if baseline_receipt.replay_cases != candidate_receipt.replay_cases:
        raise ShadowRegressionBlocked("shadow scorecards use different replay case counts")
    if baseline.total_cases != baseline_receipt.replay_cases:
        raise ShadowRegressionBlocked("baseline scorecard case count does not match its replay")
    if candidate.total_cases != candidate_receipt.replay_cases:
        raise ShadowRegressionBlocked("candidate scorecard case count does not match its replay")
    if baseline.thresholds.model_dump(mode="json") != candidate.thresholds.model_dump(mode="json"):
        raise ShadowRegressionBlocked("shadow scorecards use different review thresholds")
    if not baseline.gate_passed:
        raise ShadowRegressionBlocked("baseline scorecard must already pass its review gate")

    active = tolerance or ShadowRegressionTolerance()
    deltas = {
        "case_pass_rate_delta": _stable_delta(candidate.case_pass_rate, baseline.case_pass_rate),
        "authority_score_delta": _stable_delta(candidate.authority_score, baseline.authority_score),
        "grounding_score_delta": _stable_delta(candidate.grounding_score, baseline.grounding_score),
        "abstention_score_delta": _stable_delta(
            candidate.abstention_score,
            baseline.abstention_score,
        ),
        "privacy_score_delta": _stable_delta(candidate.privacy_score, baseline.privacy_score),
        "failed_case_delta": candidate.failed_cases - baseline.failed_cases,
        "followup_case_delta": len(candidate.followup_cases) - len(baseline.followup_cases),
    }
    reasons: list[str] = []
    if not candidate.gate_passed:
        reasons.append("candidate shadow review gate did not pass")
    for field in (
        "case_pass_rate_delta",
        "authority_score_delta",
        "grounding_score_delta",
        "abstention_score_delta",
        "privacy_score_delta",
    ):
        if deltas[field] < -active.max_score_drop:
            reasons.append(f"{field.removesuffix('_delta')} regressed beyond tolerance")
    if deltas["failed_case_delta"] > active.max_failed_case_increase:
        reasons.append("failed case count increased beyond tolerance")
    if deltas["followup_case_delta"] > active.max_followup_case_increase:
        reasons.append("follow-up case count increased beyond tolerance")

    payload = {
        "schema_version": "sentinel.shadow-regression-report.v1",
        "baseline_candidate_id": baseline.candidate_id,
        "candidate_id": candidate.candidate_id,
        "baseline_scorecard_digest": baseline.scorecard_digest,
        "candidate_scorecard_digest": candidate.scorecard_digest,
        "baseline_receipt_digest": baseline_receipt.receipt_digest,
        "candidate_receipt_digest": candidate_receipt.receipt_digest,
        "replay_sha256": baseline_receipt.replay_sha256,
        "replay_cases": baseline_receipt.replay_cases,
        "thresholds": baseline.thresholds.model_dump(mode="json"),
        "tolerance": active.model_dump(mode="json"),
        **deltas,
        "regression_reasons": reasons,
        "regression_passed": not reasons,
        "manual_review_required": True,
        "benchmark_recheck_required": True,
        "owner_decision_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowRegressionReport.model_validate(
        {**payload, "report_digest": _digest(payload)}
    )


def append_shadow_regression_history(
    report: ShadowRegressionReport,
    history: ShadowRegressionHistory | None = None,
) -> ShadowRegressionHistory:
    _require_report_digest(report)
    existing = history or _empty_history()
    _require_history_digest(existing)
    if any(item.report_digest == report.report_digest for item in existing.entries):
        raise ShadowRegressionBlocked("regression report already exists in history")
    entries = [
        *existing.entries,
        ShadowRegressionHistoryEntry(
            baseline_candidate_id=report.baseline_candidate_id,
            candidate_id=report.candidate_id,
            replay_sha256=report.replay_sha256,
            report_digest=report.report_digest,
            regression_passed=report.regression_passed,
        ),
    ]
    if len(entries) > _MAX_HISTORY:
        raise ShadowRegressionBlocked("shadow regression history exceeds its entry limit")
    payload = {
        "schema_version": "sentinel.shadow-regression-history.v1",
        "entries": [item.model_dump(mode="json") for item in entries],
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return ShadowRegressionHistory.model_validate(
        {**payload, "history_digest": _digest(payload)}
    )


def load_shadow_regression_report(path: str | Path) -> ShadowRegressionReport:
    try:
        report = ShadowRegressionReport.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid shadow regression report") from exc
    _require_report_digest(report)
    return report


def load_shadow_regression_history(path: str | Path) -> ShadowRegressionHistory:
    try:
        history = ShadowRegressionHistory.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid shadow regression history") from exc
    _require_history_digest(history)
    return history


def write_shadow_regression_report(report: ShadowRegressionReport, path: str | Path) -> None:
    _write_once(report.model_dump(mode="json"), path, "shadow regression report")


def write_shadow_regression_history(history: ShadowRegressionHistory, path: str | Path) -> None:
    _write_once(history.model_dump(mode="json"), path, "shadow regression history")


def _empty_history() -> ShadowRegressionHistory:
    payload = {
        "schema_version": "sentinel.shadow-regression-history.v1",
        "entries": [],
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
    }
    return ShadowRegressionHistory.model_validate(
        {**payload, "history_digest": _digest(payload)}
    )


def _require_scorecard_receipt_binding(
    scorecard: ShadowReviewScorecard,
    receipt: ShadowReplayReceipt,
    label: str,
) -> None:
    if scorecard.receipt_digest != receipt.receipt_digest:
        raise ShadowRegressionBlocked(f"{label} scorecard does not match supplied receipt")
    if scorecard.candidate_id != receipt.candidate_id:
        raise ShadowRegressionBlocked(f"{label} candidate does not match supplied receipt")
    if scorecard.results_sha256 != receipt.results_sha256:
        raise ShadowRegressionBlocked(f"{label} result digest does not match supplied receipt")


def _require_scorecard_digest(scorecard: ShadowReviewScorecard) -> None:
    payload = scorecard.model_dump(mode="json")
    claimed = payload.pop("scorecard_digest")
    if claimed != _digest(payload):
        raise ShadowRegressionBlocked("shadow review scorecard digest does not match contents")


def _require_receipt_digest(receipt: ShadowReplayReceipt) -> None:
    payload = receipt.model_dump(mode="json")
    claimed = payload.pop("receipt_digest")
    if claimed != _digest(payload):
        raise ShadowRegressionBlocked("shadow replay receipt digest does not match contents")


def _require_report_digest(report: ShadowRegressionReport) -> None:
    payload = report.model_dump(mode="json")
    claimed = payload.pop("report_digest")
    if claimed != _digest(payload):
        raise ShadowRegressionBlocked("shadow regression report digest does not match contents")


def _require_history_digest(history: ShadowRegressionHistory) -> None:
    payload = history.model_dump(mode="json")
    claimed = payload.pop("history_digest")
    if claimed != _digest(payload):
        raise ShadowRegressionBlocked("shadow regression history digest does not match contents")


def _write_once(payload: object, path: str | Path, label: str) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"{label} already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _stable_delta(current: float, baseline: float) -> float:
    return round(current - baseline, 12)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ShadowRegressionBlocked",
    "ShadowRegressionHistory",
    "ShadowRegressionReport",
    "ShadowRegressionTolerance",
    "append_shadow_regression_history",
    "build_shadow_regression_report",
    "load_shadow_regression_history",
    "load_shadow_regression_report",
    "write_shadow_regression_history",
    "write_shadow_regression_report",
]
