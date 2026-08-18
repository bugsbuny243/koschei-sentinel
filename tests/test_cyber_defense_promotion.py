import hashlib

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
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutEvaluationReport,
)
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    GoldHoldoutEvaluationEvidence,
)
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)
from koschei_sentinel.training import canonical_json


STAGES = [
    CyberTrainingStage.KNOWLEDGE_CONTINUED_PRETRAINING,
    CyberTrainingStage.DEFENSE_REFLEX_SFT,
    CyberTrainingStage.ADVERSARIAL_REASONING,
    CyberTrainingStage.CYBER_RANGE_REGRESSION,
]
ADAPTER_DIGEST = "9" * 64


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


def _gold_report(
    revision: str = ADAPTER_DIGEST,
    *,
    passed: bool = True,
    score: float | None = None,
) -> GoldHoldoutEvaluationReport:
    metric = score if score is not None else (1.0 if passed else 0.5)
    payload = {
        "schema_version": "sentinel.gold-holdout-evaluation-report.v1",
        "model_ref": "sentinel:candidate",
        "model_revision": revision,
        "adapter_digest": ADAPTER_DIGEST,
        "case_count": 2,
        "prediction_count": 2,
        "missing_case_ids": [],
        "extra_case_ids": [],
        "structural_exact_cases": 2 if metric == 1.0 else 1,
        "structural_exact_rate": metric,
        "compared_steps": 2,
        "predicted_steps": 2,
        "mode_accuracy": metric,
        "action_accuracy": metric,
        "target_accuracy": metric,
        "evidence_grounding_rate": 1.0,
        "target_grounding_rate": 1.0,
        "outcome_verification_rate": 1.0,
        "case_results": [],
        "passed": passed,
        "violations": [] if passed else ["fixture Gold HOLDOUT failure"],
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return GoldHoldoutEvaluationReport(**payload, report_sha256=digest)


def _policy_sha(policy: GoldHoldoutEvaluationPolicy) -> str:
    return hashlib.sha256(
        canonical_json(policy.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _gold_evidence(
    revision: str = ADAPTER_DIGEST,
    *,
    passed: bool = True,
    score: float | None = None,
    policy: GoldHoldoutEvaluationPolicy | None = None,
) -> GoldHoldoutEvaluationEvidence:
    selected_policy = policy or GoldHoldoutEvaluationPolicy()
    report = _gold_report(revision, passed=passed, score=score)
    payload = {
        "schema_version": "sentinel.gold-holdout-evaluation-evidence.v1",
        "model_ref": report.model_ref,
        "model_revision": report.model_revision,
        "adapter_digest": report.adapter_digest,
        "source_gold_audit_sha256": "1" * 64,
        "inference_inputs_sha256": "2" * 64,
        "inference_plan_sha256": "3" * 64,
        "inference_receipt_sha256": "4" * 64,
        "inference_verification_sha256": "5" * 64,
        "generation_policy_sha256": "6" * 64,
        "evaluation_policy_sha256": _policy_sha(selected_policy),
        "case_count": report.case_count,
        "prediction_count": report.prediction_count,
        "failure_count": 0,
        "inference_verification_valid": True,
        "complete_case_accounting": True,
        "report": report.model_dump(mode="json"),
        "passed": report.passed,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return GoldHoldoutEvaluationEvidence(**payload, evidence_sha256=digest)


def _evidence(
    revision: str = ADAPTER_DIGEST,
    *,
    multi_passed: bool = True,
    load_passed: bool = True,
    gold_passed: bool = True,
    gold_policy: GoldHoldoutEvaluationPolicy | None = None,
):
    policy = gold_policy or GoldHoldoutEvaluationPolicy()
    return build_cyber_defense_promotion_evidence(
        promotion_id="promotion:test",
        candidate_model_ref="sentinel:candidate",
        candidate_model_revision=revision,
        training_bundle=_bundle(),
        cyber_range_report=_single(),
        multi_incident_range_report=_multi(multi_passed),
        defense_load_range_report=_load(load_passed),
        gold_holdout_evidence=_gold_evidence(
            revision,
            passed=gold_passed,
            policy=policy,
        ),
        gold_holdout_policy=policy,
    )


def test_promotion_requires_all_four_defense_gate_families_to_pass() -> None:
    assert _evidence().ready_for_promotion is True

    multi_failed = _evidence(multi_passed=False)
    assert multi_failed.ready_for_promotion is False
    assert multi_failed.multi_incident_range_passed is False
    assert multi_failed.gold_holdout_passed is True

    load_failed = _evidence(load_passed=False)
    assert load_failed.ready_for_promotion is False
    assert load_failed.defense_load_range_passed is False
    assert load_failed.gold_holdout_passed is True

    gold_failed = _evidence(gold_passed=False)
    assert gold_failed.ready_for_promotion is False
    assert gold_failed.gold_holdout_passed is False


def test_gold_holdout_evidence_must_belong_to_candidate_revision() -> None:
    with pytest.raises(ValueError, match="model_revision differs"):
        build_cyber_defense_promotion_evidence(
            promotion_id="promotion:test",
            candidate_model_ref="sentinel:candidate",
            candidate_model_revision="8" * 64,
            training_bundle=_bundle(),
            cyber_range_report=_single(),
            multi_incident_range_report=_multi(),
            defense_load_range_report=_load(),
            gold_holdout_evidence=_gold_evidence(ADAPTER_DIGEST),
            gold_holdout_policy=GoldHoldoutEvaluationPolicy(),
        )


def test_promotion_requires_same_policy_used_to_build_gold_evidence() -> None:
    evidence = _gold_evidence(policy=GoldHoldoutEvaluationPolicy())
    stricter = GoldHoldoutEvaluationPolicy(
        minimum_structural_exact_rate=0.99,
        minimum_mode_accuracy=0.99,
        minimum_action_accuracy=0.99,
        minimum_target_accuracy=0.99,
    )

    with pytest.raises(ValueError, match="different policy"):
        build_cyber_defense_promotion_evidence(
            promotion_id="promotion:test",
            candidate_model_ref="sentinel:candidate",
            candidate_model_revision=ADAPTER_DIGEST,
            training_bundle=_bundle(),
            cyber_range_report=_single(),
            multi_incident_range_report=_multi(),
            defense_load_range_report=_load(),
            gold_holdout_evidence=evidence,
            gold_holdout_policy=stricter,
        )


def test_gold_evidence_self_hash_is_verified() -> None:
    evidence = _gold_evidence()
    tampered = evidence.model_copy(update={"inference_inputs_sha256": "7" * 64})
    with pytest.raises(ValueError, match="evidence self-hash does not verify"):
        build_cyber_defense_promotion_evidence(
            promotion_id="promotion:test",
            candidate_model_ref="sentinel:candidate",
            candidate_model_revision=ADAPTER_DIGEST,
            training_bundle=_bundle(),
            cyber_range_report=_single(),
            multi_incident_range_report=_multi(),
            defense_load_range_report=_load(),
            gold_holdout_evidence=tampered,
            gold_holdout_policy=GoldHoldoutEvaluationPolicy(),
        )


def test_candidate_revision_is_bound_into_promotion_digest() -> None:
    first = _evidence(ADAPTER_DIGEST)
    other_revision = "8" * 64
    second = build_cyber_defense_promotion_evidence(
        promotion_id="promotion:test",
        candidate_model_ref="sentinel:candidate",
        candidate_model_revision=other_revision,
        training_bundle=_bundle(),
        cyber_range_report=_single(),
        multi_incident_range_report=_multi(),
        defense_load_range_report=_load(),
        gold_holdout_evidence=_gold_evidence(other_revision),
        gold_holdout_policy=GoldHoldoutEvaluationPolicy(),
    )
    assert first.evidence_sha256 != second.evidence_sha256


def test_promotion_evidence_is_deterministic() -> None:
    first = _evidence()
    second = _evidence()
    assert first.model_dump() == second.model_dump()
    assert first.evidence_sha256 == second.evidence_sha256
    assert first.world_line_containment_rate == 1.0
    assert first.scheduler_service_coverage == 1.0
    assert first.gold_holdout_structural_exact_rate == 1.0
    assert first.gold_holdout_policy_sha256
    assert first.gold_holdout_inference_verification_sha256 == "5" * 64
    assert first.schema_version == "sentinel.cyber-defense-promotion-evidence.v4"
