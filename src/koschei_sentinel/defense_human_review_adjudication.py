from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.defense_human_review_queue import (
    HumanDefenseReviewSubmission,
    HumanDefenseReviewTask,
    verify_human_review_submission,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class HumanDefenseAdjudicationDecision(StrEnum):
    APPROVE_FOR_RANGE_VALIDATION = "APPROVE_FOR_RANGE_VALIDATION"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECT = "REJECT"


class HumanDefenseAdjudication(StrictModel):
    schema_version: Literal["sentinel.human-defense-adjudication.v1"] = (
        "sentinel.human-defense-adjudication.v1"
    )
    task_id: str
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    submission_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_id: str = Field(min_length=3, max_length=256)
    adjudicator_id: str = Field(min_length=3, max_length=256)
    decision: HumanDefenseAdjudicationDecision
    rationale: str = Field(min_length=24, max_length=8000)
    adjudication_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    approved_for_range_validation: bool
    range_outcome_verification_required: Literal[True] = True
    training_authorization: Literal[False] = False
    adjudication_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def independent_adjudicator(self) -> "HumanDefenseAdjudication":
        if self.reviewer_id == self.adjudicator_id:
            raise ValueError("reviewer and adjudicator must be different identities")
        expected = self.decision is HumanDefenseAdjudicationDecision.APPROVE_FOR_RANGE_VALIDATION
        if self.approved_for_range_validation != expected:
            raise ValueError("adjudication decision and range-validation approval disagree")
        return self


def _digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("adjudication_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def create_human_defense_adjudication(
    task: HumanDefenseReviewTask,
    submission: HumanDefenseReviewSubmission,
    *,
    adjudicator_id: str,
    decision: HumanDefenseAdjudicationDecision,
    rationale: str,
    adjudication_evidence_ids: list[str],
) -> HumanDefenseAdjudication:
    verify_human_review_submission(task, submission)
    if adjudicator_id == submission.reviewer_id:
        raise ValueError("reviewer cannot adjudicate their own defense review")

    allowed = set(task.allowed_evidence_ids)
    unknown = sorted(set(adjudication_evidence_ids) - allowed)
    if unknown:
        raise ValueError(
            "adjudication cites evidence outside the blinded task: " + ", ".join(unknown)
        )

    payload: dict[str, object] = {
        "schema_version": "sentinel.human-defense-adjudication.v1",
        "task_id": task.task_id,
        "task_sha256": task.task_sha256,
        "submission_sha256": submission.submission_sha256,
        "reviewer_id": submission.reviewer_id,
        "adjudicator_id": adjudicator_id,
        "decision": decision.value,
        "rationale": rationale,
        "adjudication_evidence_ids": list(dict.fromkeys(adjudication_evidence_ids)),
        "approved_for_range_validation": (
            decision is HumanDefenseAdjudicationDecision.APPROVE_FOR_RANGE_VALIDATION
        ),
        "range_outcome_verification_required": True,
        "training_authorization": False,
    }
    payload["adjudication_sha256"] = _digest(payload)
    return HumanDefenseAdjudication.model_validate(payload)


def verify_human_defense_adjudication(
    task: HumanDefenseReviewTask,
    submission: HumanDefenseReviewSubmission,
    adjudication: HumanDefenseAdjudication,
) -> None:
    verify_human_review_submission(task, submission)
    if adjudication.task_id != task.task_id or adjudication.task_sha256 != task.task_sha256:
        raise ValueError("adjudication is not bound to the supplied review task")
    if adjudication.submission_sha256 != submission.submission_sha256:
        raise ValueError("adjudication is not bound to the supplied review submission")
    if adjudication.reviewer_id != submission.reviewer_id:
        raise ValueError("adjudication reviewer identity differs from submission")
    if adjudication.adjudicator_id == submission.reviewer_id:
        raise ValueError("reviewer cannot adjudicate their own defense review")
    if _digest(adjudication.model_dump(mode="json")) != adjudication.adjudication_sha256:
        raise ValueError("human defense adjudication self-hash does not verify")
    allowed = set(task.allowed_evidence_ids)
    unknown = sorted(set(adjudication.adjudication_evidence_ids) - allowed)
    if unknown:
        raise ValueError(
            "adjudication references evidence outside the blinded task: "
            + ", ".join(unknown)
        )
