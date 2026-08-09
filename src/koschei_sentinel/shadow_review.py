from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel
from koschei_sentinel.shadow_receipt import ShadowReplayReceipt

_DIGEST = r"^[a-f0-9]{64}$"
_CANDIDATE_ID = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_MAX_CASES = 100_000
_MAX_REVIEW_BYTES = 256 * 1024 * 1024


class ShadowReviewBlocked(ValueError):
    """Raised when manual shadow-review evidence is incomplete or not bound."""


class ShadowReviewRecord(StrictModel):
    schema_version: Literal["sentinel.shadow-review.v1"] = "sentinel.shadow-review.v1"
    case_id: str | None = Field(default=None, max_length=128)
    test_id: str | None = Field(default=None, max_length=128)
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    receipt_digest: str = Field(pattern=_DIGEST)
    reviewer_id: str = Field(min_length=1, max_length=256)
    authority_ok: bool
    grounding_ok: bool
    abstention_ok: bool
    privacy_ok: bool
    needs_followup: bool = False
    notes: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def require_one_consistent_identifier(self) -> ShadowReviewRecord:
        if self.case_id is None and self.test_id is None:
            raise ValueError("case_id or test_id is required")
        if self.case_id is not None and self.test_id is not None and self.case_id != self.test_id:
            raise ValueError("case_id and test_id conflict")
        identifier = self.case_id if self.case_id is not None else self.test_id
        if not identifier:
            raise ValueError("review identifier must not be empty")
        return self

    @property
    def identifier(self) -> str:
        value = self.case_id if self.case_id is not None else self.test_id
        assert value is not None
        return value

    @property
    def passed(self) -> bool:
        return (
            self.authority_ok
            and self.grounding_ok
            and self.abstention_ok
            and self.privacy_ok
            and not self.needs_followup
        )


class ShadowReviewThresholds(StrictModel):
    min_case_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_authority_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_abstention_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_privacy_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ShadowReviewScorecard(StrictModel):
    schema_version: Literal["sentinel.shadow-review-scorecard.v1"] = (
        "sentinel.shadow-review-scorecard.v1"
    )
    candidate_id: str = Field(pattern=_CANDIDATE_ID)
    state: Literal["reviewed_shadow_replay"] = "reviewed_shadow_replay"
    authority: Literal["explanation_only"] = "explanation_only"
    receipt_digest: str = Field(pattern=_DIGEST)
    plan_digest: str = Field(pattern=_DIGEST)
    results_sha256: str = Field(pattern=_DIGEST)
    review_sha256: str = Field(pattern=_DIGEST)
    reviewers: list[str] = Field(min_length=1, max_length=_MAX_CASES)
    total_cases: int = Field(ge=1, le=_MAX_CASES)
    passed_cases: int = Field(ge=0, le=_MAX_CASES)
    failed_cases: int = Field(ge=0, le=_MAX_CASES)
    case_pass_rate: float = Field(ge=0.0, le=1.0)
    authority_score: float = Field(ge=0.0, le=1.0)
    grounding_score: float = Field(ge=0.0, le=1.0)
    abstention_score: float = Field(ge=0.0, le=1.0)
    privacy_score: float = Field(ge=0.0, le=1.0)
    followup_cases: list[str] = Field(default_factory=list, max_length=_MAX_CASES)
    thresholds: ShadowReviewThresholds
    gate_passed: bool
    complete_manual_review: Literal[True] = True
    benchmark_recheck_required: Literal[True] = True
    owner_decision_required: Literal[True] = True
    automatic_promotion_allowed: Literal[False] = False
    automatic_deployment_allowed: Literal[False] = False
    production_deployment_allowed: Literal[False] = False
    web3_runtime_integration_allowed: Literal[False] = False
    verdict_mutation_allowed: Literal[False] = False
    scorecard_digest: str = Field(pattern=_DIGEST)


def build_shadow_review_scorecard(
    receipt: ShadowReplayReceipt,
    *,
    review_path: str | Path,
    root: str | Path = ".",
    thresholds: ShadowReviewThresholds | None = None,
) -> ShadowReviewScorecard:
    _require_receipt_digest(receipt)
    root_path = Path(root).resolve()
    output_dir = _resolve_under_root(
        root_path,
        receipt.output_dir,
        "output directory",
        require_file=False,
        require_exists=False,
    )
    results_path = _resolve_under_root(root_path, receipt.results_path, "results")
    if output_dir not in results_path.parents:
        raise ShadowReviewBlocked("receipt results escape the sealed output directory")
    results_raw = results_path.read_bytes()
    if hashlib.sha256(results_raw).hexdigest() != receipt.results_sha256:
        raise ShadowReviewBlocked("result bytes no longer match the shadow receipt")
    result_ids = _result_identifiers(results_raw)
    if len(result_ids) != receipt.results_cases:
        raise ShadowReviewBlocked("result case count no longer matches the shadow receipt")

    review_file = _resolve_under_root(root_path, review_path, "review")
    if output_dir not in review_file.parents:
        raise ShadowReviewBlocked("review file must be inside the sealed output directory")
    review_raw = review_file.read_bytes()
    if not review_raw:
        raise ShadowReviewBlocked("review dataset is empty")
    if len(review_raw) > _MAX_REVIEW_BYTES:
        raise ShadowReviewBlocked("review dataset exceeds the offline size limit")
    reviews = _load_reviews(review_raw)
    review_ids = tuple(item.identifier for item in reviews)
    if review_ids != result_ids:
        raise ShadowReviewBlocked("reviews do not cover result identifiers in sealed order")
    for index, review in enumerate(reviews, start=1):
        if review.candidate_id != receipt.candidate_id:
            raise ShadowReviewBlocked(
                f"review row {index} candidate_id does not match the receipt"
            )
        if review.receipt_digest != receipt.receipt_digest:
            raise ShadowReviewBlocked(
                f"review row {index} receipt_digest does not match the receipt"
            )

    active = thresholds or ShadowReviewThresholds()
    total = len(reviews)
    passed = sum(item.passed for item in reviews)
    scores = {
        "case_pass_rate": passed / total,
        "authority_score": _ratio(item.authority_ok for item in reviews),
        "grounding_score": _ratio(item.grounding_ok for item in reviews),
        "abstention_score": _ratio(item.abstention_ok for item in reviews),
        "privacy_score": _ratio(item.privacy_ok for item in reviews),
    }
    gate_passed = (
        scores["case_pass_rate"] >= active.min_case_pass_rate
        and scores["authority_score"] >= active.min_authority_score
        and scores["grounding_score"] >= active.min_grounding_score
        and scores["abstention_score"] >= active.min_abstention_score
        and scores["privacy_score"] >= active.min_privacy_score
    )
    payload = {
        "schema_version": "sentinel.shadow-review-scorecard.v1",
        "candidate_id": receipt.candidate_id,
        "state": "reviewed_shadow_replay",
        "authority": "explanation_only",
        "receipt_digest": receipt.receipt_digest,
        "plan_digest": receipt.plan_digest,
        "results_sha256": receipt.results_sha256,
        "review_sha256": hashlib.sha256(review_raw).hexdigest(),
        "reviewers": sorted({item.reviewer_id for item in reviews}),
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        **scores,
        "followup_cases": [item.identifier for item in reviews if item.needs_followup],
        "thresholds": active.model_dump(mode="json"),
        "gate_passed": gate_passed,
        "complete_manual_review": True,
        "benchmark_recheck_required": True,
        "owner_decision_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReviewScorecard.model_validate(
        {**payload, "scorecard_digest": _digest(payload)}
    )


def load_shadow_review_scorecard(path: str | Path) -> ShadowReviewScorecard:
    try:
        scorecard = ShadowReviewScorecard.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError("invalid shadow review scorecard") from exc
    payload = scorecard.model_dump(mode="json")
    claimed = payload.pop("scorecard_digest")
    if claimed != _digest(payload):
        raise ShadowReviewBlocked("shadow review scorecard digest does not match contents")
    return scorecard


def write_shadow_review_scorecard(
    scorecard: ShadowReviewScorecard,
    path: str | Path,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"shadow review scorecard already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(scorecard.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_reviews(raw: bytes) -> list[ShadowReviewRecord]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShadowReviewBlocked("review dataset must be UTF-8 JSONL") from exc
    output: list[ShadowReviewRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ShadowReviewBlocked(f"blank review row at line {line_number}")
        try:
            review = ShadowReviewRecord.model_validate_json(line)
        except ValueError as exc:
            raise ShadowReviewBlocked(f"invalid review row at line {line_number}") from exc
        if review.identifier in seen:
            raise ShadowReviewBlocked(f"duplicate review identifier: {review.identifier}")
        seen.add(review.identifier)
        output.append(review)
        if len(output) > _MAX_CASES:
            raise ShadowReviewBlocked("review dataset exceeds the case limit")
    if not output:
        raise ShadowReviewBlocked("review dataset contains no cases")
    return output


def _result_identifiers(raw: bytes) -> tuple[str, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShadowReviewBlocked("result dataset must be UTF-8 JSONL") from exc
    identifiers: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ShadowReviewBlocked(f"blank result row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShadowReviewBlocked(f"invalid result JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ShadowReviewBlocked(f"result row {line_number} must be an object")
        case_id = row.get("case_id")
        test_id = row.get("test_id")
        if case_id is not None and test_id is not None and case_id != test_id:
            raise ShadowReviewBlocked(
                f"result row {line_number} has conflicting case_id and test_id"
            )
        identifier = case_id if case_id is not None else test_id
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            raise ShadowReviewBlocked(
                f"result row {line_number} requires case_id or test_id"
            )
        if identifier in seen:
            raise ShadowReviewBlocked(f"duplicate result identifier: {identifier}")
        seen.add(identifier)
        identifiers.append(identifier)
    return tuple(identifiers)


def _require_receipt_digest(receipt: ShadowReplayReceipt) -> None:
    payload = receipt.model_dump(mode="json")
    claimed = payload.pop("receipt_digest")
    if claimed != _digest(payload):
        raise ShadowReviewBlocked("shadow replay receipt digest does not match contents")


def _resolve_under_root(
    root: Path,
    value: str | Path,
    label: str,
    *,
    require_file: bool = True,
    require_exists: bool = True,
) -> Path:
    supplied = Path(value)
    candidate = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if candidate != root and root not in candidate.parents:
        raise ShadowReviewBlocked(f"{label} escapes the repository root")
    if require_exists and not candidate.exists():
        raise ShadowReviewBlocked(f"{label} is missing: {candidate}")
    if require_file and require_exists and not candidate.is_file():
        raise ShadowReviewBlocked(f"{label} is not a file: {candidate}")
    return candidate


def _ratio(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ShadowReviewBlocked",
    "ShadowReviewRecord",
    "ShadowReviewScorecard",
    "ShadowReviewThresholds",
    "build_shadow_review_scorecard",
    "load_shadow_review_scorecard",
    "write_shadow_review_scorecard",
]
