from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_megatron_gold_evidence import (
    CyberMegatronGoldEvaluationEvidence,
    build_cyber_megatron_gold_evidence,
    verify_cyber_megatron_gold_evidence,
)
from koschei_sentinel.cyber_megatron_training import (
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
)
from koschei_sentinel.cyber_range_suite import CyberRangeSuiteReport
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle
from koschei_sentinel.defense_load_range import DefenseLoadRangeReport
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutEvaluationReport,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)
from koschei_sentinel.training import canonical_json

_DIGEST = r"^[a-f0-9]{64}$"
_PRODUCTION_MINIMUM_HOLDOUT_CASES = 50


def _sha256_canonical(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _training_bundle_sha256(bundle: CyberTrainingBundle) -> str:
    payload = "|".join(
        [
            bundle.bundle_id,
            bundle.foundation_model_ref,
            bundle.foundation_model_revision,
            bundle.knowledge_batch_id,
            bundle.knowledge_training_corpus_sha256,
            bundle.knowledge_artifact_manifest_sha256,
            bundle.defense_reflex_examples_sha256,
            str(bundle.defense_reflex_example_count),
            bundle.causal_defense_examples_sha256,
            str(bundle.causal_defense_example_count),
            bundle.eval_holdout_sha256,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _gold_passes_policy(
    report: GoldHoldoutEvaluationReport,
    policy: GoldHoldoutEvaluationPolicy,
) -> bool:
    if report.case_count < policy.minimum_case_count:
        return False
    if report.missing_case_ids or report.extra_case_ids or report.violations:
        return False
    checks = (
        (report.structural_exact_rate, policy.minimum_structural_exact_rate),
        (report.mode_accuracy, policy.minimum_mode_accuracy),
        (report.action_accuracy, policy.minimum_action_accuracy),
        (report.target_accuracy, policy.minimum_target_accuracy),
        (
            report.evidence_selection_accuracy,
            policy.minimum_evidence_selection_accuracy,
        ),
        (report.evidence_grounding_rate, policy.minimum_evidence_grounding_rate),
        (report.target_grounding_rate, policy.minimum_target_grounding_rate),
        (report.outcome_verification_rate, policy.minimum_outcome_verification_rate),
    )
    return all(observed + 1e-12 >= minimum for observed, minimum in checks)


class CyberDefensePromotionEvidenceV5(StrictModel):
    schema_version: Literal["sentinel.cyber-defense-promotion-evidence.v5"] = (
        "sentinel.cyber-defense-promotion-evidence.v5"
    )
    promotion_id: str = Field(min_length=3, max_length=256)
    candidate_model_ref: Literal["Qwen/Qwen3.5-397B-A17B"]
    candidate_model_revision: Literal["8472618112abcbd45acbcdc58436aff4233c23f7"]
    candidate_checkpoint_sha256: str = Field(pattern=_DIGEST)
    candidate_manifest_sha256: str = Field(pattern=_DIGEST)
    candidate_sha256: str = Field(pattern=_DIGEST)
    training_bundle_sha256: str = Field(pattern=_DIGEST)
    eval_holdout_sha256: str = Field(pattern=_DIGEST)
    cyber_range_suite_sha256: str = Field(pattern=_DIGEST)
    multi_incident_range_suite_sha256: str = Field(pattern=_DIGEST)
    defense_load_range_sha256: str = Field(pattern=_DIGEST)
    gold_evaluation_evidence_sha256: str = Field(pattern=_DIGEST)
    gold_evaluation_report_sha256: str = Field(pattern=_DIGEST)
    gold_evaluation_policy_sha256: str = Field(pattern=_DIGEST)
    gold_source_audit_sha256: str = Field(pattern=_DIGEST)
    gold_review_signature_audit_sha256: str = Field(pattern=_DIGEST)
    gold_holdout_pack_proof_sha256: str = Field(pattern=_DIGEST)
    gold_reviewer_trust_policy_digest: str = Field(pattern=_DIGEST)
    gold_owner_key_fingerprint: str = Field(pattern=_DIGEST)
    gold_holdout_plan_sha256: str = Field(pattern=_DIGEST)
    gold_worker_plan_sha256: str = Field(pattern=_DIGEST)
    gold_execution_state_sha256: str = Field(pattern=_DIGEST)
    gold_inference_receipt_sha256: str = Field(pattern=_DIGEST)
    gold_output_verification_sha256: str = Field(pattern=_DIGEST)
    gold_minimum_case_count: int = Field(ge=_PRODUCTION_MINIMUM_HOLDOUT_CASES)
    gold_case_count: int = Field(ge=_PRODUCTION_MINIMUM_HOLDOUT_CASES)
    gold_prediction_count: int = Field(ge=0)
    gold_failure_count: int = Field(ge=0)
    cyber_range_passed: bool
    multi_incident_range_passed: bool
    defense_load_range_passed: bool
    gold_holdout_passed: bool
    malicious_containment_rate: float = Field(ge=0.0, le=1.0)
    reroute_detection_rate: float = Field(ge=0.0, le=1.0)
    benign_high_impact_false_positive_rate: float = Field(ge=0.0, le=1.0)
    component_count_accuracy: float = Field(ge=0.0, le=1.0)
    world_line_transition_accuracy: float = Field(ge=0.0, le=1.0)
    component_containment_rate: float = Field(ge=0.0, le=1.0)
    world_line_containment_rate: float = Field(ge=0.0, le=1.0)
    mean_world_line_containment_tick_latency: float | None = Field(default=None, ge=0.0)
    cut_point_leakage_count: int = Field(ge=0)
    scheduler_service_coverage: float = Field(ge=0.0, le=1.0)
    scheduler_critical_max_first_service_wave: int | None = Field(default=None, ge=0)
    scheduler_max_wait_cycles: int = Field(ge=0)
    scheduler_capacity_violation_count: int = Field(ge=0)
    gold_holdout_structural_exact_rate: float = Field(ge=0.0, le=1.0)
    gold_holdout_mode_accuracy: float = Field(ge=0.0, le=1.0)
    gold_holdout_action_accuracy: float = Field(ge=0.0, le=1.0)
    gold_holdout_target_accuracy: float = Field(ge=0.0, le=1.0)
    gold_holdout_evidence_selection_accuracy: float = Field(ge=0.0, le=1.0)
    gold_holdout_evidence_grounding_rate: float = Field(ge=0.0, le=1.0)
    gold_holdout_target_grounding_rate: float = Field(ge=0.0, le=1.0)
    gold_holdout_outcome_verification_rate: float = Field(ge=0.0, le=1.0)
    ready_for_promotion: bool
    evidence_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def promotion_contract_verifies(self) -> CyberDefensePromotionEvidenceV5:
        if self.gold_prediction_count + self.gold_failure_count != self.gold_case_count:
            raise ValueError("Promotion v5 Gold case accounting is incomplete")
        expected_ready = (
            self.cyber_range_passed
            and self.multi_incident_range_passed
            and self.defense_load_range_passed
            and self.gold_holdout_passed
        )
        if self.ready_for_promotion != expected_ready:
            raise ValueError("Promotion v5 readiness must equal all required defense gates")
        unsigned = self.model_dump(mode="json")
        observed = str(unsigned.pop("evidence_sha256"))
        if _sha256_canonical(unsigned) != observed:
            raise ValueError("Promotion v5 evidence self-hash does not verify")
        return self


def verify_cyber_defense_promotion_v5(
    evidence: CyberDefensePromotionEvidenceV5,
) -> None:
    payload = evidence.model_dump(mode="json")
    observed = str(payload.pop("evidence_sha256"))
    if _sha256_canonical(payload) != observed:
        raise ValueError("Promotion v5 evidence self-hash does not verify")


def build_cyber_defense_promotion_v5(
    *,
    promotion_id: str,
    candidate_model_ref: str,
    candidate_model_revision: str,
    candidate_checkpoint_sha256: str,
    training_bundle: CyberTrainingBundle,
    cyber_range_report: CyberRangeSuiteReport,
    multi_incident_range_report: MultiIncidentCyberRangeSuiteReport,
    defense_load_range_report: DefenseLoadRangeReport,
    gold_evidence: CyberMegatronGoldEvaluationEvidence,
    gold_policy: GoldHoldoutEvaluationPolicy,
) -> CyberDefensePromotionEvidenceV5:
    if candidate_model_ref != QWEN35_397B_MODEL:
        raise ValueError("Promotion v5 accepts only the active Qwen3.5-397B-A17B target")
    if candidate_model_revision != QWEN35_397B_REVISION:
        raise ValueError("Promotion v5 candidate revision must equal the pinned model revision")
    if not training_bundle.ready_for_training:
        raise ValueError("Promotion v5 training bundle is not ready")
    if training_bundle.bundle_sha256 != _training_bundle_sha256(training_bundle):
        raise ValueError("Promotion v5 training bundle self-binding digest does not verify")
    if training_bundle.foundation_model_ref != candidate_model_ref:
        raise ValueError("Promotion v5 training bundle model differs from candidate")
    if training_bundle.foundation_model_revision != candidate_model_revision:
        raise ValueError("Promotion v5 training bundle revision differs from candidate")
    if gold_policy.minimum_case_count < _PRODUCTION_MINIMUM_HOLDOUT_CASES:
        raise ValueError("Promotion v5 Gold policy must require at least 50 HOLDOUT cases")

    verify_cyber_megatron_gold_evidence(gold_evidence)
    if gold_evidence.model != candidate_model_ref:
        raise ValueError("Promotion v5 Gold evidence model differs from candidate")
    if gold_evidence.model_revision != candidate_model_revision:
        raise ValueError("Promotion v5 Gold evidence revision differs from candidate")
    if gold_evidence.checkpoint_tree_sha256 != candidate_checkpoint_sha256:
        raise ValueError("Promotion v5 Gold evidence checkpoint differs from candidate")
    if gold_evidence.minimum_case_count < _PRODUCTION_MINIMUM_HOLDOUT_CASES:
        raise ValueError("Promotion v5 Gold evidence weakened the production case floor")

    policy_sha = _sha256_canonical(gold_policy.model_dump(mode="json"))
    if gold_evidence.evaluation_policy_sha256 != policy_sha:
        raise ValueError("Promotion v5 Gold evidence was evaluated under a different policy")
    report = gold_evidence.report
    gold_passed = (
        gold_evidence.passed
        and gold_evidence.failure_count == 0
        and _gold_passes_policy(report, gold_policy)
    )

    single_sha = _sha256_canonical(cyber_range_report.model_dump(mode="json"))
    multi_sha = _sha256_canonical(multi_incident_range_report.model_dump(mode="json"))
    load_sha = _sha256_canonical(defense_load_range_report.model_dump(mode="json"))
    ready = (
        cyber_range_report.passed
        and multi_incident_range_report.passed
        and defense_load_range_report.passed
        and gold_passed
    )

    unsigned: dict[str, object] = {
        "schema_version": "sentinel.cyber-defense-promotion-evidence.v5",
        "promotion_id": promotion_id,
        "candidate_model_ref": candidate_model_ref,
        "candidate_model_revision": candidate_model_revision,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "candidate_manifest_sha256": gold_evidence.candidate_manifest_sha256,
        "candidate_sha256": gold_evidence.candidate_sha256,
        "training_bundle_sha256": training_bundle.bundle_sha256,
        "eval_holdout_sha256": training_bundle.eval_holdout_sha256,
        "cyber_range_suite_sha256": single_sha,
        "multi_incident_range_suite_sha256": multi_sha,
        "defense_load_range_sha256": load_sha,
        "gold_evaluation_evidence_sha256": gold_evidence.evidence_sha256,
        "gold_evaluation_report_sha256": report.report_sha256,
        "gold_evaluation_policy_sha256": policy_sha,
        "gold_source_audit_sha256": gold_evidence.source_gold_audit_sha256,
        "gold_review_signature_audit_sha256": (
            gold_evidence.review_signature_audit_sha256
        ),
        "gold_holdout_pack_proof_sha256": gold_evidence.pack_proof_sha256,
        "gold_reviewer_trust_policy_digest": (
            gold_evidence.reviewer_trust_policy_digest
        ),
        "gold_owner_key_fingerprint": gold_evidence.owner_key_fingerprint,
        "gold_holdout_plan_sha256": gold_evidence.holdout_plan_sha256,
        "gold_worker_plan_sha256": gold_evidence.worker_plan_sha256,
        "gold_execution_state_sha256": gold_evidence.execution_state_sha256,
        "gold_inference_receipt_sha256": gold_evidence.inference_receipt_sha256,
        "gold_output_verification_sha256": gold_evidence.output_verification_sha256,
        "gold_minimum_case_count": gold_evidence.minimum_case_count,
        "gold_case_count": gold_evidence.case_count,
        "gold_prediction_count": gold_evidence.prediction_count,
        "gold_failure_count": gold_evidence.failure_count,
        "cyber_range_passed": cyber_range_report.passed,
        "multi_incident_range_passed": multi_incident_range_report.passed,
        "defense_load_range_passed": defense_load_range_report.passed,
        "gold_holdout_passed": gold_passed,
        "malicious_containment_rate": cyber_range_report.malicious_containment_rate,
        "reroute_detection_rate": cyber_range_report.reroute_detection_rate,
        "benign_high_impact_false_positive_rate": (
            cyber_range_report.benign_high_impact_false_positive_rate
        ),
        "component_count_accuracy": multi_incident_range_report.component_count_accuracy,
        "world_line_transition_accuracy": (
            multi_incident_range_report.world_line_transition_accuracy
        ),
        "component_containment_rate": (
            multi_incident_range_report.component_containment_rate
        ),
        "world_line_containment_rate": (
            multi_incident_range_report.world_line_containment_rate
        ),
        "mean_world_line_containment_tick_latency": (
            multi_incident_range_report.mean_world_line_containment_tick_latency
        ),
        "cut_point_leakage_count": multi_incident_range_report.cut_point_leakage_count,
        "scheduler_service_coverage": defense_load_range_report.service_coverage,
        "scheduler_critical_max_first_service_wave": (
            defense_load_range_report.critical_max_first_service_wave
        ),
        "scheduler_max_wait_cycles": defense_load_range_report.max_wait_cycles,
        "scheduler_capacity_violation_count": (
            defense_load_range_report.capacity_violation_count
        ),
        "gold_holdout_structural_exact_rate": report.structural_exact_rate,
        "gold_holdout_mode_accuracy": report.mode_accuracy,
        "gold_holdout_action_accuracy": report.action_accuracy,
        "gold_holdout_target_accuracy": report.target_accuracy,
        "gold_holdout_evidence_selection_accuracy": report.evidence_selection_accuracy,
        "gold_holdout_evidence_grounding_rate": report.evidence_grounding_rate,
        "gold_holdout_target_grounding_rate": report.target_grounding_rate,
        "gold_holdout_outcome_verification_rate": report.outcome_verification_rate,
        "ready_for_promotion": ready,
    }
    return CyberDefensePromotionEvidenceV5(
        **unsigned,
        evidence_sha256=_sha256_canonical(unsigned),
    )


def build_cyber_defense_promotion_v5_from_sources(
    *,
    promotion_id: str,
    candidate_model_ref: str,
    candidate_model_revision: str,
    candidate_checkpoint_sha256: str,
    training_bundle: CyberTrainingBundle,
    cyber_range_report: CyberRangeSuiteReport,
    multi_incident_range_report: MultiIncidentCyberRangeSuiteReport,
    defense_load_range_report: DefenseLoadRangeReport,
    supplied_gold_evidence: CyberMegatronGoldEvaluationEvidence,
    gold_policy: GoldHoldoutEvaluationPolicy,
    gold_release_dir: str | Path,
    gold_worker_dir: str | Path,
    gold_holdout_plan_path: str | Path,
    gold_candidate_manifest_path: str | Path,
    gold_candidate_config_path: str | Path,
    gold_candidate_training_plan_path: str | Path,
    gold_checkpoint_dir: str | Path,
    gold_inference_pack: str | Path,
    gold_pack_signature: str | Path,
    gold_reviewer_public_key: str | Path,
    gold_reviewer_trust_policy: str | Path,
    gold_owner_public_key: str | Path,
    minimum_case_count: int = _PRODUCTION_MINIMUM_HOLDOUT_CASES,
    root: str | Path = ".",
) -> CyberDefensePromotionEvidenceV5:
    fresh_gold = build_cyber_megatron_gold_evidence(
        root=root,
        release_dir=gold_release_dir,
        worker_dir=gold_worker_dir,
        holdout_plan_path=gold_holdout_plan_path,
        candidate_manifest_path=gold_candidate_manifest_path,
        candidate_config_path=gold_candidate_config_path,
        candidate_training_plan_path=gold_candidate_training_plan_path,
        checkpoint_dir=gold_checkpoint_dir,
        inference_pack=gold_inference_pack,
        signature_path=gold_pack_signature,
        reviewer_public_key_path=gold_reviewer_public_key,
        reviewer_trust_policy_path=gold_reviewer_trust_policy,
        owner_public_key_path=gold_owner_public_key,
        policy=gold_policy,
        minimum_case_count=minimum_case_count,
    )
    if supplied_gold_evidence.model_dump(mode="json") != fresh_gold.model_dump(mode="json"):
        raise ValueError(
            "supplied 397B Gold evidence differs from fresh source-artifact rebuild"
        )
    return build_cyber_defense_promotion_v5(
        promotion_id=promotion_id,
        candidate_model_ref=candidate_model_ref,
        candidate_model_revision=candidate_model_revision,
        candidate_checkpoint_sha256=candidate_checkpoint_sha256,
        training_bundle=training_bundle,
        cyber_range_report=cyber_range_report,
        multi_incident_range_report=multi_incident_range_report,
        defense_load_range_report=defense_load_range_report,
        gold_evidence=fresh_gold,
        gold_policy=gold_policy,
    )
