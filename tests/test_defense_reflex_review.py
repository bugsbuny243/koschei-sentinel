import pytest

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_reflex_candidates import (
    DefenseReflexCandidate,
    DefenseReflexFailureType,
    DefenseReflexReviewStatus,
)
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionStep,
    review_defense_reflex_candidate,
)


def _candidate() -> DefenseReflexCandidate:
    return DefenseReflexCandidate(
        candidate_id="reflex:scenario:no-containment",
        scenario_id="scenario",
        failure_type=DefenseReflexFailureType.NO_CONTAINMENT,
        source_report_sha256="a" * 64,
        failing_ticks=[0, 1],
        observed_modes=["GUARD", "COMBAT"],
        attempted_actions=["REVOKE_CREDENTIAL"],
        rationale=["fixture"],
        review_status=DefenseReflexReviewStatus.REVIEW_REQUIRED,
        training_authorization=False,
    )


def _step() -> ReviewedCorrectionStep:
    return ReviewedCorrectionStep(
        sequence=1,
        expected_mode=DefenseMode.COMBAT,
        action=DefenseActionType.REVOKE_CREDENTIAL,
        target_entity_id="cred:compromised",
        rationale="Revoke the corroborated compromised credential before further progression.",
        supporting_evidence_ids=["evidence:credential-use", "evidence:credential-theft"],
        outcome_verification_ids=["evidence:revocation-confirmed"],
    )


def test_approved_correction_requires_verified_outcome() -> None:
    with pytest.raises(ValueError, match="verified outcome"):
        review_defense_reflex_candidate(
            _candidate(),
            reviewer_id="reviewer:security",
            decision=CorrectionReviewDecision.APPROVE,
            corrected_interpretation="The two corroborated observations establish malicious credential use.",
            corrected_steps=[_step()],
            review_evidence_ids=["review:one"],
            outcome_verified=False,
            authorize_for_training=True,
        )


def test_review_requires_evidence() -> None:
    with pytest.raises(ValueError, match="review evidence"):
        review_defense_reflex_candidate(
            _candidate(),
            reviewer_id="reviewer:security",
            decision=CorrectionReviewDecision.REJECT,
            corrected_interpretation="The candidate was caused by an invalid range expectation and is rejected.",
            corrected_steps=[],
            review_evidence_ids=[],
            outcome_verified=False,
        )


def test_approved_verified_correction_can_be_training_authorized_deterministically() -> None:
    kwargs = dict(
        reviewer_id="reviewer:security",
        decision=CorrectionReviewDecision.APPROVE,
        corrected_interpretation="The evidence supports Combat mode and immediate credential revocation.",
        corrected_steps=[_step()],
        review_evidence_ids=["review:one", "review:two"],
        outcome_verified=True,
        authorize_for_training=True,
    )
    first = review_defense_reflex_candidate(_candidate(), **kwargs)
    second = review_defense_reflex_candidate(_candidate(), **kwargs)

    assert first.training_authorization is True
    assert first.review_status is DefenseReflexReviewStatus.APPROVED
    assert first.correction_sha256 == second.correction_sha256
    assert first.correction_id == second.correction_id
