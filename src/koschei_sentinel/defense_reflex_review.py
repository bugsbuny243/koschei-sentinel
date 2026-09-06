from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_reflex_candidates import (
    DefenseReflexCandidate,
    DefenseReflexReviewStatus,
)
from koschei_sentinel.models import StrictModel


class CorrectionReviewDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class ReviewedCorrectionStep(StrictModel):
    sequence: int = Field(ge=1)
    expected_mode: DefenseMode
    action: DefenseActionType
    target_entity_id: str = Field(min_length=2, max_length=256)
    rationale: str = Field(min_length=8, max_length=4000)
    supporting_evidence_ids: list[str] = Field(min_length=1, max_length=128)
    outcome_verification_ids: list[str] = Field(min_length=1, max_length=128)


class ReviewedCorrectionTrajectory(StrictModel):
    schema_version: Literal["sentinel.defense-reflex-correction.v1"] = (
        "sentinel.defense-reflex-correction.v1"
    )
    correction_id: str
    candidate_id: str
    scenario_id: str
    failure_type: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_id: str = Field(min_length=3, max_length=256)
    review_decision: CorrectionReviewDecision
    corrected_interpretation: str = Field(min_length=16, max_length=8000)
    corrected_steps: list[ReviewedCorrectionStep] = Field(default_factory=list, max_length=64)
    review_evidence_ids: list[str] = Field(min_length=1, max_length=256)
    review_status: DefenseReflexReviewStatus
    outcome_verified: bool
    training_authorization: bool
    correction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def fail_closed_training_authorization(self) -> ReviewedCorrectionTrajectory:
        if self.review_decision is CorrectionReviewDecision.APPROVE:
            if self.review_status is not DefenseReflexReviewStatus.APPROVED:
                raise ValueError("approved correction must carry APPROVED review status")
            if not self.corrected_steps:
                raise ValueError("approved correction requires at least one corrected step")
            if not self.outcome_verified:
                raise ValueError("approved correction requires verified outcome evidence")
        if self.training_authorization:
            if self.review_decision is not CorrectionReviewDecision.APPROVE:
                raise ValueError("training authorization requires an approved review decision")
            if self.review_status is not DefenseReflexReviewStatus.APPROVED:
                raise ValueError("training authorization requires APPROVED review status")
            if not self.outcome_verified:
                raise ValueError("training authorization requires a verified correction outcome")
        if self.review_decision is CorrectionReviewDecision.REJECT and self.training_authorization:
            raise ValueError("rejected corrections cannot be training-authorized")
        return self


def _payload_digest(
    *,
    candidate: DefenseReflexCandidate,
    reviewer_id: str,
    review_decision: CorrectionReviewDecision,
    corrected_interpretation: str,
    corrected_steps: list[ReviewedCorrectionStep],
    review_evidence_ids: list[str],
    outcome_verified: bool,
) -> str:
    payload = "|".join(
        [
            candidate.candidate_id,
            candidate.source_report_sha256,
            reviewer_id,
            review_decision.value,
            corrected_interpretation,
            ",".join(review_evidence_ids),
            str(int(outcome_verified)),
            ";".join(
                f"{step.sequence}:{step.expected_mode.value}:{step.action.value}:"
                f"{step.target_entity_id}:{','.join(step.supporting_evidence_ids)}:"
                f"{','.join(step.outcome_verification_ids)}"
                for step in corrected_steps
            ),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def review_defense_reflex_candidate(
    candidate: DefenseReflexCandidate,
    *,
    reviewer_id: str,
    decision: CorrectionReviewDecision,
    corrected_interpretation: str,
    corrected_steps: list[ReviewedCorrectionStep],
    review_evidence_ids: list[str],
    outcome_verified: bool,
    authorize_for_training: bool = False,
) -> ReviewedCorrectionTrajectory:
    if candidate.review_status is DefenseReflexReviewStatus.REJECTED:
        raise ValueError("rejected candidate cannot be reviewed into a correction trajectory")
    if not review_evidence_ids:
        raise ValueError("review evidence is required")

    sequences = [step.sequence for step in corrected_steps]
    if sequences and sequences != list(range(1, len(sequences) + 1)):
        raise ValueError("corrected steps must be contiguous and ordered from 1")

    if decision is CorrectionReviewDecision.REJECT:
        if authorize_for_training:
            raise ValueError("rejected correction cannot be training-authorized")
        review_status = DefenseReflexReviewStatus.REJECTED
        outcome_verified = False
    else:
        review_status = DefenseReflexReviewStatus.APPROVED
        if not corrected_steps:
            raise ValueError("approved correction requires corrected defensive steps")
        if not outcome_verified:
            raise ValueError("approved correction requires verified outcome")

    digest = _payload_digest(
        candidate=candidate,
        reviewer_id=reviewer_id,
        review_decision=decision,
        corrected_interpretation=corrected_interpretation,
        corrected_steps=corrected_steps,
        review_evidence_ids=review_evidence_ids,
        outcome_verified=outcome_verified,
    )

    return ReviewedCorrectionTrajectory(
        correction_id=f"correction:{candidate.candidate_id}:{digest[:16]}",
        candidate_id=candidate.candidate_id,
        scenario_id=candidate.scenario_id,
        failure_type=candidate.failure_type.value,
        source_report_sha256=candidate.source_report_sha256,
        reviewer_id=reviewer_id,
        review_decision=decision,
        corrected_interpretation=corrected_interpretation,
        corrected_steps=corrected_steps,
        review_evidence_ids=list(dict.fromkeys(review_evidence_ids)),
        review_status=review_status,
        outcome_verified=outcome_verified,
        training_authorization=(
            authorize_for_training
            and decision is CorrectionReviewDecision.APPROVE
            and outcome_verified
        ),
        correction_sha256=digest,
    )
