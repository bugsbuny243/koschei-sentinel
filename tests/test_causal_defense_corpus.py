import pytest

from koschei_sentinel.causal_defense_corpus import build_causal_defense_example
from koschei_sentinel.cyber_range import CyberRangeScenario, ScenarioTruth, run_cyber_range_scenario
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.cyber_world_model_episode import build_world_model_episode
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


def _episode():
    graph = CyberStateGraph(
        graph_id="causal:incident",
        entities=[
            CyberEntity(entity_id="cred:a", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:a", entity_type=CyberEntityType.DEVICE),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:credential",
                source_entity_id="cred:a",
                target_entity_id="device:a",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[
                    EvidenceRef(
                        evidence_id="ev:credential",
                        source="causal-fixture",
                        content_sha256="a" * 64,
                    )
                ],
            )
        ],
    )
    scenario = CyberRangeScenario(
        scenario_id="causal-scenario",
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[graph],
        critical_entity_ids=["device:a"],
    )
    report = run_cyber_range_scenario(scenario)
    return build_world_model_episode(scenario, report)


def _correction(source_sha: str):
    candidate = DefenseReflexCandidate(
        candidate_id="reflex:causal:no-containment",
        scenario_id="causal-scenario",
        failure_type=DefenseReflexFailureType.NO_CONTAINMENT,
        source_report_sha256=source_sha,
        failing_ticks=[0],
        observed_modes=["COMBAT"],
        attempted_actions=["REVOKE_CREDENTIAL"],
        rationale=["fixture"],
        review_status=DefenseReflexReviewStatus.REVIEW_REQUIRED,
        training_authorization=False,
    )
    step = ReviewedCorrectionStep(
        sequence=1,
        expected_mode=DefenseMode.COMBAT,
        action=DefenseActionType.REVOKE_CREDENTIAL,
        target_entity_id="cred:a",
        rationale="Revoke the corroborated compromised credential before further progression.",
        supporting_evidence_ids=["ev:credential"],
        outcome_verification_ids=["ev:revoked"],
    )
    return review_defense_reflex_candidate(
        candidate,
        reviewer_id="reviewer:causal",
        decision=CorrectionReviewDecision.APPROVE,
        corrected_interpretation="Observed malicious credential use requires verified active containment.",
        corrected_steps=[step],
        review_evidence_ids=["review:causal"],
        outcome_verified=True,
        authorize_for_training=True,
    )


def test_causal_example_requires_same_source_report_digest() -> None:
    episode = _episode()
    with pytest.raises(ValueError, match="source report digest"):
        build_causal_defense_example(episode, _correction("f" * 64))


def test_causal_example_binds_temporal_episode_to_reviewed_correction() -> None:
    episode = _episode()
    example = build_causal_defense_example(
        episode,
        _correction(episode.source_report_sha256),
    )

    assert example.world_model_episode_sha256 == episode.episode_sha256
    assert example.source_report_sha256 == episode.source_report_sha256
    assert example.expected_defense_sequence[0]["action"] == "REVOKE_CREDENTIAL"
    assert example.temporal_snapshots[0]["observed_relations"] == 1
