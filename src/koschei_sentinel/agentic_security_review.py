from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.agentic_security_eval_generator import GeneratedBundle
from koschei_sentinel.models import StrictModel

Decision = Literal["accepted", "rejected", "needs_rewrite"]


class ReviewRecord(StrictModel):
    schema_version: Literal["sentinel.agentic-security-review-record.v1"] = "sentinel.agentic-security-review-record.v1"
    case_id: str = Field(min_length=1)
    source_lineage_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Decision
    reviewer_id: str = Field(min_length=1, max_length=256)
    reviewer_evidence_ref: str = Field(min_length=1, max_length=2048)
    rationale: str = Field(min_length=1, max_length=8000)
    rewritten_case_ref: str | None = Field(default=None, max_length=2048)
    record_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def rewrite_decision_requires_reference(self) -> "ReviewRecord":
        if self.decision == "needs_rewrite" and not self.rewritten_case_ref:
            raise ValueError("needs_rewrite requires rewritten_case_ref")
        if self.decision != "needs_rewrite" and self.rewritten_case_ref is not None:
            raise ValueError("rewritten_case_ref is only valid for needs_rewrite")
        return self


class ReviewBundle(StrictModel):
    schema_version: Literal["sentinel.agentic-security-review-bundle.v1"] = "sentinel.agentic-security-review-bundle.v1"
    source_benchmark_schema: Literal["sentinel.agentic-security-generated-benchmark.v1"]
    source_benchmark_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    records: list[ReviewRecord] = Field(min_length=1)
    accepted_are_gold: Literal[False] = False
    automatic_gold_promotion_allowed: Literal[False] = False

    @model_validator(mode="after")
    def unique_case_reviews(self) -> "ReviewBundle":
        ids = [record.case_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("review bundle cannot contain duplicate case decisions")
        return self


def _sha(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generated_bundle_digest(bundle: GeneratedBundle) -> str:
    return _sha(bundle.model_dump(mode="json"))


def review_record_digest(record: ReviewRecord) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("record_sha256", None)
    return _sha(payload)


def validate_review_bundle(source: GeneratedBundle, review: ReviewBundle) -> None:
    if review.source_benchmark_schema != source.schema_version:
        raise ValueError("review source schema does not match generated benchmark")
    if review.source_benchmark_sha256 != generated_bundle_digest(source):
        raise ValueError("review source benchmark digest does not match generated benchmark")

    cases = {case.case_id: case for case in source.cases}
    for record in review.records:
        case = cases.get(record.case_id)
        if case is None:
            raise ValueError(f"review references unknown case: {record.case_id}")
        if record.source_lineage_sha256 != case.lineage_sha256:
            raise ValueError(f"review lineage mismatch for case: {record.case_id}")
        if record.record_sha256 != review_record_digest(record):
            raise ValueError(f"review record self-hash mismatch for case: {record.case_id}")


def load_generated_bundle(path: Path) -> GeneratedBundle:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load generated benchmark from {path}") from exc
    return GeneratedBundle.model_validate(payload)


def load_review_bundle(path: Path) -> ReviewBundle:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load review bundle from {path}") from exc
    return ReviewBundle.model_validate(payload)


def load_and_validate_review(source_path: Path, review_path: Path) -> tuple[GeneratedBundle, ReviewBundle]:
    source = load_generated_bundle(source_path)
    review = load_review_bundle(review_path)
    validate_review_bundle(source, review)
    return source, review
