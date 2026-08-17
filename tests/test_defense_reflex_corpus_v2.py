import hashlib

import pytest

from koschei_sentinel.cyber_range import CyberRangeScenario, ScenarioTruth, run_cyber_range_scenario
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.defense_reflex_candidates import DefenseReflexReviewStatus
from koschei_sentinel.defense_reflex_corpus_v2 import build_defense_reflex_v2_example
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionStep,
    ReviewedCorrectionTrajectory,
)


def _scenario(confidence: float = 0.8) -> CyberRangeScenario:
    graph = CyberStateGraph(
        graph_id="incident:reflex-v2-test",
        entities=[
            CyberEntity(entity_id="credential:test", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:test", entity_type=CyberEntityType.DEVICE),
        ],
        relations=[
            CyberRelation(
                relation_id="relation:credential-use",
                source_entity_id="credential:test",
                target_entity_id="device:test",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=confidence,
                evidence=[
                    EvidenceRef(
                        evidence_id="evidence:credential-use",
                        source="test-sensor",
                        content_sha256="a" * 64,
                    )
                ],
            )
        ],
    )
    return CyberRangeScenario(
        scenario_id="scenario:reflex-v2-test",
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[graph],
        critical_entity_ids=["device:test"],
    )


def _report_sha(scenario: CyberRangeScenario) -> str:
    report = run_cyber_range_scenario(scenario)
    return hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()


def _correction(scenario: CyberRangeScenario) -> ReviewedCorrectionTrajectory:
    return ReviewedCorrectionTrajectory(
        correction_id="correction:reflex-v2-test",
        candidate_id="candidate:reflex-v2-test",
        scenario_id=scenario.scenario_id,
        failure_type="NO_CONTAINMENT",
        source_report_sha256=_report_sha(scenario),
        reviewer_id="reviewer:test",
        review_decision=CorrectionReviewDecision.APPROVE,
        corrected_interpretation=(
            "Observed credential use on the protected device requires evidence-backed containment."
        ),
        corrected_steps=[
            ReviewedCorrectionStep(
                sequence=1,
                expected_mode=DefenseMode.COMBAT,
                action=DefenseActionType.REVOKE_CREDENTIAL,
                target_entity_id="credential:test",
                rationale="Invalidate the credential used on the protected attack path.",
                supporting_evidence_ids=["evidence:credential-use"],
                outcome_verification_ids=["outcome:credential-revoked"],
            )
        ],
        review_evidence_ids=["review:evidence:1"],
        review_status=DefenseReflexReviewStatus.APPROVED,
        outcome_verified=True,
        training_authorization=True,
        correction_sha256="b" * 64,
    )


def test_v2_example_binds_graph_timeline_to_reviewed_report() -> None:
    scenario = _scenario()
    correction = _correction(scenario)
    example = build_defense_reflex_v2_example(scenario, correction)

    assert example.source_report_sha256 == correction.source_report_sha256
    assert example.graph_snapshots[0]["graph_id"] == "incident:reflex-v2-test"
    assert example.corrected_interpretation == correction.corrected_interpretation
    assert example.expected_sequence[0]["action"] == "REVOKE_CREDENTIAL"


def test_v2_rejects_scenario_drift_after_review() -> None:
    reviewed_scenario = _scenario(confidence=0.8)
    correction = _correction(reviewed_scenario)
    changed_scenario = _scenario(confidence=0.9)

    with pytest.raises(ValueError, match="no longer reproduces"):
        build_defense_reflex_v2_example(changed_scenario, correction)
