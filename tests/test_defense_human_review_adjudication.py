from __future__ import annotations

import pytest

from koschei_sentinel.defense_authority import DefenseAction, DefenseMode
from koschei_sentinel.defense_human_review_adjudication import (
    HumanDefenseAdjudicationDecision,
    create_human_defense_adjudication,
    verify_human_defense_adjudication,
)
from koschei_sentinel.defense_human_review_queue import (
    HumanDefenseReviewStep,
    build_human_review_task,
    create_human_review_submission,
)
from tests.test_defense_human_review_queue import _example


def _review():
    task = build_human_review_task(_example())
    submission = create_human_review_submission(
        task,
        reviewer_id="reviewer:one",
        interpretation=(
            "The graph supports additional evidence collection before stronger containment authority."
        ),
        steps=[
            HumanDefenseReviewStep(
                sequence=1,
                expected_mode=DefenseMode.GUARD,
                action=DefenseAction.COLLECT_EVIDENCE,
                target_entity_id="endpoint:1",
                rationale="Collect evidence while preserving the current endpoint state.",
                supporting_evidence_ids=["ev-1"],
            )
        ],
        review_evidence_ids=["ev-1"],
        training_authorization_requested=True,
    )
    return task, submission


def test_independent_adjudication_approves_only_range_validation() -> None:
    task, submission = _review()
    adjudication = create_human_defense_adjudication(
        task,
        submission,
        adjudicator_id="adjudicator:two",
        decision=HumanDefenseAdjudicationDecision.APPROVE_FOR_RANGE_VALIDATION,
        rationale=(
            "The proposed Guard action is supported by the cited evidence and should be range tested."
        ),
        adjudication_evidence_ids=["ev-1"],
    )

    verify_human_defense_adjudication(task, submission, adjudication)
    assert adjudication.approved_for_range_validation is True
    assert adjudication.range_outcome_verification_required is True
    assert adjudication.training_authorization is False


def test_reviewer_cannot_adjudicate_own_submission() -> None:
    task, submission = _review()

    with pytest.raises(ValueError, match="cannot adjudicate"):
        create_human_defense_adjudication(
            task,
            submission,
            adjudicator_id=submission.reviewer_id,
            decision=HumanDefenseAdjudicationDecision.APPROVE_FOR_RANGE_VALIDATION,
            rationale="The reviewer must not be permitted to approve their own defense submission.",
            adjudication_evidence_ids=["ev-1"],
        )


def test_adjudication_rejects_evidence_outside_blinded_task() -> None:
    task, submission = _review()

    with pytest.raises(ValueError, match="outside the blinded task"):
        create_human_defense_adjudication(
            task,
            submission,
            adjudicator_id="adjudicator:two",
            decision=HumanDefenseAdjudicationDecision.NEEDS_REVISION,
            rationale=(
                "The proposed review needs revision and cannot rely on evidence absent from the task."
            ),
            adjudication_evidence_ids=["invented-evidence"],
        )
