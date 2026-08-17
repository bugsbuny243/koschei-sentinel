from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_range_suite import CyberRangeSuiteReport
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle
from koschei_sentinel.models import StrictModel
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeSuiteReport,
)


_DIGEST = r"^[a-f0-9]{64}$"


class CyberDefensePromotionEvidence(StrictModel):
    schema_version: Literal["sentinel.cyber-defense-promotion-evidence.v1"] = (
        "sentinel.cyber-defense-promotion-evidence.v1"
    )
    promotion_id: str = Field(min_length=3, max_length=256)
    candidate_model_ref: str = Field(min_length=3, max_length=512)
    candidate_model_revision: str = Field(min_length=3, max_length=256)
    training_bundle_sha256: str = Field(pattern=_DIGEST)
    eval_holdout_sha256: str = Field(pattern=_DIGEST)
    cyber_range_suite_sha256: str = Field(pattern=_DIGEST)
    multi_incident_range_suite_sha256: str = Field(pattern=_DIGEST)
    cyber_range_passed: bool
    multi_incident_range_passed: bool
    malicious_containment_rate: float = Field(ge=0.0, le=1.0)
    reroute_detection_rate: float = Field(ge=0.0, le=1.0)
    benign_high_impact_false_positive_rate: float = Field(ge=0.0, le=1.0)
    component_count_accuracy: float = Field(ge=0.0, le=1.0)
    world_line_transition_accuracy: float = Field(ge=0.0, le=1.0)
    component_containment_rate: float = Field(ge=0.0, le=1.0)
    world_line_containment_rate: float = Field(ge=0.0, le=1.0)
    mean_world_line_containment_tick_latency: float | None = Field(default=None, ge=0.0)
    cut_point_leakage_count: int = Field(ge=0)
    ready_for_promotion: bool
    evidence_sha256: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def promotion_is_fail_closed(self) -> "CyberDefensePromotionEvidence":
        expected = self.cyber_range_passed and self.multi_incident_range_passed
        if self.ready_for_promotion != expected:
            raise ValueError("promotion readiness must equal all required range gates")
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
) -> CyberDefensePromotionEvidence:
    if not training_bundle.ready_for_training:
        raise ValueError("cyber training bundle is not ready")

    single_sha = _stable_sha(cyber_range_report)
    multi_sha = _stable_sha(multi_incident_range_report)
    ready = cyber_range_report.passed and multi_incident_range_report.passed
    digest_payload = "|".join(
        [
            promotion_id,
            candidate_model_ref,
            candidate_model_revision,
            training_bundle.bundle_sha256,
            training_bundle.eval_holdout_sha256,
            single_sha,
            multi_sha,
            str(int(cyber_range_report.passed)),
            str(int(multi_incident_range_report.passed)),
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
        cyber_range_passed=cyber_range_report.passed,
        multi_incident_range_passed=multi_incident_range_report.passed,
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
        ready_for_promotion=ready,
        evidence_sha256=digest,
    )
