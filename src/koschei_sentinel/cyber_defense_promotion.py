from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_range_suite import CyberRangeSuiteReport
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle
from koschei_sentinel.defense_load_range import DefenseLoadRangeReport
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationReport
from koschei_sentinel.models import StrictModel
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)


_DIGEST = r"^[a-f0-9]{64}$"


class CyberDefensePromotionEvidence(StrictModel):
    schema_version: Literal["sentinel.cyber-defense-promotion-evidence.v3"] = (
        "sentinel.cyber-defense-promotion-evidence.v3"
    )
    promotion_id: str = Field(min_length=3, max_length=256)
    candidate_model_ref: str = Field(min_length=3, max_length=512)
    candidate_model_revision: str = Field(min_length=3, max_length=256)
    training_bundle_sha256: str = Field(pattern=_DIGEST)
    eval_holdout_sha256: str = Field(pattern=_DIGEST)
    cyber_range_suite_sha256: str = Field(pattern=_DIGEST)
    multi_incident_range_suite_sha256: str = Field(pattern=_DIGEST)
    defense_load_range_sha256: str = Field(pattern=_DIGEST)
    gold_holdout_evaluation_sha256: str = Field(pattern=_DIGEST)
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
    gold_holdout_evidence_grounding_rate: float = Field(ge=0.0, le=1.0)
    gold_holdout_target_grounding_rate: float = Field(ge=0.0, le=1.0)
    gold_holdout_outcome_verification_rate: float = Field(ge=0.0, le=1.0)
    ready_for_promotion: bool
    evidence_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def promotion_is_fail_closed(self) -> "CyberDefensePromotionEvidence":
        expected = (
            self.cyber_range_passed
            and self.multi_incident_range_passed
            and self.defense_load_range_passed
            and self.gold_holdout_passed
        )
        if self.ready_for_promotion != expected:
            raise ValueError("promotion readiness must equal all required defense gates")
        return self


def _stable_sha(value: StrictModel) -> str:
    payload = json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cyber_defense_promotion_evidence(
    *,
    promotion_id: str,
    candidate_model_ref: str,
    candidate_model_revision: str,
    training_bundle: CyberTrainingBundle,
    cyber_range_report: CyberRangeSuiteReport,
    multi_incident_range_report: MultiIncidentCyberRangeSuiteReport,
    defense_load_range_report: DefenseLoadRangeReport,
    gold_holdout_report: GoldHoldoutEvaluationReport,
) -> CyberDefensePromotionEvidence:
    if not training_bundle.ready_for_training:
        raise ValueError("cyber training bundle is not ready")
    if gold_holdout_report.model_ref != candidate_model_ref:
        raise ValueError("Gold HOLDOUT report model_ref differs from promotion candidate")
    if gold_holdout_report.model_revision != candidate_model_revision:
        raise ValueError("Gold HOLDOUT report model_revision differs from promotion candidate")

    single_sha = _stable_sha(cyber_range_report)
    multi_sha = _stable_sha(multi_incident_range_report)
    load_sha = _stable_sha(defense_load_range_report)
    gold_sha = _stable_sha(gold_holdout_report)
    ready = (
        cyber_range_report.passed
        and multi_incident_range_report.passed
        and defense_load_range_report.passed
        and gold_holdout_report.passed
    )
    digest_payload = "|".join(
        [
            promotion_id,
            candidate_model_ref,
            candidate_model_revision,
            training_bundle.bundle_sha256,
            training_bundle.eval_holdout_sha256,
            single_sha,
            multi_sha,
            load_sha,
            gold_sha,
            str(int(cyber_range_report.passed)),
            str(int(multi_incident_range_report.passed)),
            str(int(defense_load_range_report.passed)),
            str(int(gold_holdout_report.passed)),
        ]
    )
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()

    return CyberDefensePromotionEvidence(
        promotion_id=promotion_id,
        candidate_model_ref=candidate_model_ref,
        candidate_model_revision=candidate_model_revision,
        training_bundle_sha256=training_bundle.bundle_sha256,
        eval_holdout_sha256=training_bundle.eval_holdout_sha256,
        cyber_range_suite_sha256=single_sha,
        multi_incident_range_suite_sha256=multi_sha,
        defense_load_range_sha256=load_sha,
        gold_holdout_evaluation_sha256=gold_sha,
        cyber_range_passed=cyber_range_report.passed,
        multi_incident_range_passed=multi_incident_range_report.passed,
        defense_load_range_passed=defense_load_range_report.passed,
        gold_holdout_passed=gold_holdout_report.passed,
        malicious_containment_rate=cyber_range_report.malicious_containment_rate,
        reroute_detection_rate=cyber_range_report.reroute_detection_rate,
        benign_high_impact_false_positive_rate=(
            cyber_range_report.benign_high_impact_false_positive_rate
        ),
        component_count_accuracy=multi_incident_range_report.component_count_accuracy,
        world_line_transition_accuracy=(
            multi_incident_range_report.world_line_transition_accuracy
        ),
        component_containment_rate=(
            multi_incident_range_report.component_containment_rate
        ),
        world_line_containment_rate=(
            multi_incident_range_report.world_line_containment_rate
        ),
        mean_world_line_containment_tick_latency=(
            multi_incident_range_report.mean_world_line_containment_tick_latency
        ),
        cut_point_leakage_count=multi_incident_range_report.cut_point_leakage_count,
        scheduler_service_coverage=defense_load_range_report.service_coverage,
        scheduler_critical_max_first_service_wave=(
            defense_load_range_report.critical_max_first_service_wave
        ),
        scheduler_max_wait_cycles=defense_load_range_report.max_wait_cycles,
        scheduler_capacity_violation_count=(
            defense_load_range_report.capacity_violation_count
        ),
        gold_holdout_structural_exact_rate=gold_holdout_report.structural_exact_rate,
        gold_holdout_mode_accuracy=gold_holdout_report.mode_accuracy,
        gold_holdout_action_accuracy=gold_holdout_report.action_accuracy,
        gold_holdout_target_accuracy=gold_holdout_report.target_accuracy,
        gold_holdout_evidence_grounding_rate=gold_holdout_report.evidence_grounding_rate,
        gold_holdout_target_grounding_rate=gold_holdout_report.target_grounding_rate,
        gold_holdout_outcome_verification_rate=(
            gold_holdout_report.outcome_verification_rate
        ),
        ready_for_promotion=ready,
        evidence_sha256=digest,
    )
