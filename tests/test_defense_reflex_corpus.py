import pytest

from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_reflex_candidates import (
    DefenseReflexCandidate,
    DefenseReflexFailureType,
    DefenseReflexReviewStatus,
)
from koschei_sentinel.defense_reflex_corpus import (
    build_defense_reflex_examples,
    build_manifest,
    serialize_examples,
)
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionStep,
    review_defense_reflex_candidate,
)


def _candidate(name: str) -> DefenseReflexCandidate:
    return DefenseReflexCandidate(
        candidate_id=f"reflex:{name}:no-containment",
        scenario_id=name,
        failure_type=DefenseReflexFailureType.NO_CONTAINMENT,
        source_report_sha256=("a" if name == "one" else "b") * 64,
        failing_ticks=[0],
        observed_modes=["COMBAT"],
        attempted_actions=["REVOKE_CREDENTIAL"],
        rationale=["fixture"],
        review_status=DefenseReflexReviewStatus.REVIEW_REQUIRED,
        training_authorization=False,
    )


def _approved(name: str, authorize: bool = True):
    step = ReviewedCorrectionStep(
        sequence=1,
        expected_mode=DefenseMode.COMBAT,
        action=DefenseActionType.REVOKE_CREDENTIAL,
        target_entity_id=f"cred:{name}",
        rationale="Revoke the corroborated compromised credential to interrupt progression.",
        supporting_evidence_ids=[f"evidence:{name}:pre"],
        outcome_verification_ids=[f"evidence:{name}:post"],
    )
    return review_defense_reflex_candidate(
        _candidate(name),
        reviewer_id="reviewer:security",
        decision=CorrectionReviewDecision.APPROVE,
        corrected_interpretation="Corroborated malicious credential use requires active containment.",
        corrected_steps=[step],
        review_evidence_ids=[f"review:{name}"],
        outcome_verified=True,
        authorize_for_training=authorize,
    )


def test_any_unauthorized_correction_fails_the_entire_export() -> None:
    with pytest.raises(ValueError, match="lacks training authorization"):
        build_defense_reflex_examples([_approved("one"), _approved("two", authorize=False)])


def test_authorized_export_is_deterministic() -> None:
    corrections = [_approved("two"), _approved("one")]
    first = build_defense_reflex_examples(corrections)
    second = build_defense_reflex_examples(list(reversed(corrections)))

    assert serialize_examples(first) == serialize_examples(second)
    assert build_manifest(first).examples_sha256 == build_manifest(second).examples_sha256
    assert build_manifest(first).ready_for_training_pipeline is True
