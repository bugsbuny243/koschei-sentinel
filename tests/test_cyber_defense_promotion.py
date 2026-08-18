import pytest

from koschei_sentinel.cyber_defense_promotion import (
    build_cyber_defense_promotion_evidence,
)
from koschei_sentinel.cyber_range_suite import CyberRangeSuiteReport
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle, CyberTrainingStage
from koschei_sentinel.defense_load_range import (
    DefenseLoadGatePolicy,
    DefenseLoadRangeReport,
)
from koschei_sentinel.defense_resource_scheduler import (
    DefenseResourceClass,
    DefenseResourcePolicy,
)
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationReport
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)


STAGES = [
    CyberTrainingStage.KNOWLEDGE_CONTINUED_PRETRAINING,
    CyberTrainingStage.DEFENSE_REFLEX_SFT,
    CyberTrainingStage.ADVERSARIAL_REASONING,
    CyberTrainingStage.CYBER_RANGE_REGRESSION,
]


def _bundle() -> CyberTrainingBundle:
    return CyberTrainingBundle(
        bundle_id="bundle:promotion",
        foundation_model_ref="foundation:test",
        foundation_model_revision="foundation-revision:test",
        knowledge_batch_id="batch:test",
        knowledge_training_corpus_sha256="a" * 64,
        knowledge_artifact_manifest_sha256="b" * 64,
        defense_reflex_examples_sha256="c" * 64,
        defense_reflex_example_count=10,
        causal_defense_examples_sha256="d" * 64,
        causal_defense_example_count=10,
        eval_holdout_sha256="e" * 64,
        required_stages=STAGES,
        ready_for_training=True,
        bundle_sha256="f" * 64,
    )


def _single(passed: bool = True) -> CyberRangeSuiteReport:
    return CyberRangeSuiteReport(
        scenarios=1,
        malicious_scenarios=1,
        benign_scenarios=0,
        malicious_contained=1 if passed else 0,
        malicious_containment_rate=1.0 if passed else 0.0,
        reroutes_expected=1,
        reroutes_detected=1 if passed else 0,
        reroute_detection_rate=1.0 if passed else 0.0,
        benign_with_high_impact_false_positive=0,
        benign_high_impact_false_positive_rate=0.0,
        mean_containment_tick=1.0 if passed else None,
        passed=passed,
        violations=[] if passed else ["fixture failure"],
        scenario_reports=[],
    )


def _multi(passed: bool = True) -> MultiIncidentCyberRangeSuiteReport:
    return MultiIncidentCyberRangeSuiteReport(
        scenarios=1,
        component_count_accuracy=1.0 if passed else 0.5,
        world_line_transition_accuracy=1.0 if passed else 0.5,
        total_component_instances=2,
        contained_component_instances=2 if passed else 1,
        component_containment_rate=1.0 if passed else 0.5,
        total_world_lines=2,
        contained_world_lines=2 if passed else 1,
        world_line_containment_rate=1.0 if passed else 0.5,
        cut_point_leakage_count=0,
        mean_containment_step=1.0,
        mean_world_line_containment_tick_latency=0.0 if passed else 1.0,
        passed=passed,
        violations=[] if passed else ["fixture failure"],
        scenario_reports=[],
    )


def _load(passed: bool = True) -> DefenseLoadRangeReport:
    return DefenseLoadRangeReport(
        graph_id="incident:load",
        resource_policy=DefenseResourcePolicy(),
        gate_policy=DefenseLoadGatePolicy(),
        eligible_components=2,
        critical_eligible_components=1,
        serviced_components=2 if passed else 1,
        service_coverage=1.0 if passed else 0.5,
        waves_run=2,
        mean_first_service_wave=0.5 if passed else 0.0,
        max_first_service_wave=1 if passed else 0,
        critical_max_first_service_wave=0,
        max_wait_cycles=1 if passed else 9,
        starved_component_ids=[] if passed else ["component:starved"],
        critical_starved_component_ids=[],
        max_resource_utilization={resource: 0.0 for resource in DefenseResourceClass},
        capacity_violation_count=0,
        passed=passed,
        violations=[] if passed else ["fixture load failure"],
        waves=[],
    )


def _gold(
    revision: str = "candidate-revision:1",
    *,
    passed: bool = True,
) -> GoldHoldoutEvaluationReport:
    score = 1.0 if passed else 0.5
    return GoldHoldoutEvaluationReport(
        model_ref="sentinel:candidate",
        model_revision=revision,
        adapter_digest="9" * 64,
        case_count=2,
        prediction_count=2,
        missing_case_ids=[],
        extra_case_ids=[],
        structural_exact_cases=2 if passed else 1,
        structural_exact_rate=score,
        compared_steps=2,
        predicted_steps=2,
        mode_accuracy=score,
        action_accuracy=score,
        target_accuracy=score,
        evidence_grounding_rate=1.0,
        target_grounding_rate=1.0,
        outcome_verification_rate=1.0,
        case_results=[],
        passed=passed,
        violations=[] if passed else ["fixture Gold HOLDOUT failure"],
        report_sha256="8" * 64,
    )


def _evidence(
    revision: str = "candidate-revision:1",
    *,
    multi_passed: bool = True,
    load_passed: bool = True,
    gold_passed: bool = True,
):
    return build_cyber_defense_promotion_evidence(
        promotion_id="promotion:test",
        candidate_model_ref="sentinel:candidate",
        candidate_model_revision=revision,
        training_bundle=_bundle(),
        cyber_range_report=_single(),
        multi_incident_range_report=_multi(multi_passed),
        defense_load_range_report=_load(load_passed),
        gold_holdout_report=_gold(revision, passed=gold_passed),
    )


def test_promotion_requires_all_four_defense_gate_families_to_pass() -> None:
    assert _evidence().ready_for_promotion is True

    multi_failed = _evidence(multi_passed=False)
    assert multi_failed.ready_for_promotion is False
    assert multi_failed.cyber_range_passed is True
    assert multi_failed.multi_incident_range_passed is False
    assert multi_failed.defense_load_range_passed is True
    assert multi_failed.gold_holdout_passed is True

    load_failed = _evidence(load_passed=False)
    assert load_failed.ready_for_promotion is False
    assert load_failed.cyber_range_passed is True
    assert load_failed.multi_incident_range_passed is True
    assert load_failed.defense_load_range_passed is False
    assert load_failed.gold_holdout_passed is True

    gold_failed = _evidence(gold_passed=False)
    assert gold_failed.ready_for_promotion is False
    assert gold_failed.cyber_range_passed is True
    assert gold_failed.multi_incident_range_passed is True
    assert gold_failed.defense_load_range_passed is True
    assert gold_failed.gold_holdout_passed is False


def test_gold_holdout_report_must_belong_to_candidate_revision() -> None:
    with pytest.raises(ValueError, match="model_revision differs"):
        build_cyber_defense_promotion_evidence(
            promotion_id="promotion:test",
            candidate_model_ref="sentinel:candidate",
            candidate_model_revision="candidate-revision:2",
            training_bundle=_bundle(),
            cyber_range_report=_single(),
            multi_incident_range_report=_multi(),
            defense_load_range_report=_load(),
            gold_holdout_report=_gold("candidate-revision:1"),
        )


def test_candidate_revision_is_bound_into_promotion_digest() -> None:
    first = _evidence("candidate-revision:1")
    second = _evidence("candidate-revision:2")
    assert first.evidence_sha256 != second.evidence_sha256


def test_promotion_evidence_is_deterministic() -> None:
    first = _evidence()
    second = _evidence()
    assert first.model_dump() == second.model_dump()
    assert first.evidence_sha256 == second.evidence_sha256
    assert first.world_line_containment_rate == 1.0
    assert first.scheduler_service_coverage == 1.0
    assert first.gold_holdout_structural_exact_rate == 1.0
    assert first.schema_version == "sentinel.cyber-defense-promotion-evidence.v3"
