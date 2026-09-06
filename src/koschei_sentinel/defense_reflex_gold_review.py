from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_range import CyberRangeScenario, run_cyber_range_scenario
from koschei_sentinel.defense_reflex_gold_queue import (
    GoldDefenseReviewPacket,
    GoldReviewSplit,
    _packet_digest,
)
from koschei_sentinel.defense_reflex_review import (
    CorrectionReviewDecision,
    ReviewedCorrectionStep,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.training import canonical_json


class GoldHumanReviewSpec(StrictModel):
    schema_version: Literal["sentinel.gold-human-review-spec.v1"] = (
        "sentinel.gold-human-review-spec.v1"
    )
    reviewer_id: str = Field(min_length=3, max_length=256)
    decision: CorrectionReviewDecision
    interpretation: str = Field(min_length=16, max_length=8000)
    expected_steps: list[ReviewedCorrectionStep] = Field(default_factory=list, max_length=64)
    review_evidence_ids: list[str] = Field(default_factory=list, max_length=256)
    outcome_verified: bool = False
    authorize_for_training: bool = False
    authorize_for_evaluation: bool = False

    @model_validator(mode="after")
    def decision_is_coherent(self) -> GoldHumanReviewSpec:
        if self.decision is CorrectionReviewDecision.APPROVE:
            if not self.expected_steps:
                raise ValueError("approved Gold review requires expected defensive steps")
            if not self.review_evidence_ids:
                raise ValueError("approved Gold review requires review evidence")
            if not self.outcome_verified:
                raise ValueError("approved Gold review requires verified outcome evidence")
        else:
            if self.authorize_for_training or self.authorize_for_evaluation:
                raise ValueError("rejected Gold review cannot authorize training or evaluation")
            if self.expected_steps:
                raise ValueError("rejected Gold review must not carry expected defensive steps")
        return self


class GoldReviewedPacket(StrictModel):
    schema_version: Literal["sentinel.gold-reviewed-packet.v1"] = (
        "sentinel.gold-reviewed-packet.v1"
    )
    packet_id: str
    packet_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scenario_id: str
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    split: GoldReviewSplit
    reviewer_id: str
    decision: CorrectionReviewDecision
    interpretation: str
    expected_steps: list[ReviewedCorrectionStep]
    review_evidence_ids: list[str]
    outcome_verified: bool
    training_authorization: bool
    evaluation_authorization: bool
    promotion_eligible: bool
    review_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def authorization_is_fail_closed(self) -> GoldReviewedPacket:
        if self.decision is CorrectionReviewDecision.REJECT:
            if self.training_authorization or self.evaluation_authorization:
                raise ValueError("rejected Gold packet cannot carry authorization")
            if self.promotion_eligible:
                raise ValueError("rejected Gold packet cannot be promotion-eligible")
            return self
        if not self.outcome_verified or not self.expected_steps:
            raise ValueError("approved Gold packet requires verified expected defense")
        if self.split is GoldReviewSplit.HOLDOUT:
            if self.training_authorization:
                raise ValueError("Gold HOLDOUT packet can never authorize training")
            if not self.evaluation_authorization:
                raise ValueError("approved Gold HOLDOUT packet requires evaluation authorization")
            if self.promotion_eligible:
                raise ValueError("Gold HOLDOUT packet is evaluation-only, not training-promotion data")
        else:
            if not self.training_authorization:
                raise ValueError("approved Gold TRAIN/VALIDATION packet requires training authorization")
            if not self.promotion_eligible:
                raise ValueError("human-reviewed training packet must be promotion-eligible")
        return self


def _scenario_report_sha256(scenario: CyberRangeScenario) -> str:
    report = run_cyber_range_scenario(scenario)
    return hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()


def _scenario_entity_ids(scenario: CyberRangeScenario) -> set[str]:
    return {
        entity.entity_id
        for graph in scenario.graph_snapshots
        for entity in graph.entities
    }


def _scenario_evidence_ids(scenario: CyberRangeScenario) -> set[str]:
    return {
        evidence.evidence_id
        for graph in scenario.graph_snapshots
        for relation in graph.relations
        for evidence in relation.evidence
    }


def _review_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("review_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def review_gold_packet(
    packet: GoldDefenseReviewPacket,
    scenario: CyberRangeScenario,
    spec: GoldHumanReviewSpec,
) -> GoldReviewedPacket:
    if _packet_digest(packet.model_dump(mode="json")) != packet.packet_sha256:
        raise ValueError("Gold review packet self-hash does not verify")
    if packet.scenario_id != scenario.scenario_id:
        raise ValueError("Gold review packet and scenario IDs differ")
    report_sha = _scenario_report_sha256(scenario)
    if report_sha != packet.source_report_sha256:
        raise ValueError("Gold review scenario drifted after queue assignment")

    entity_ids = _scenario_entity_ids(scenario)
    evidence_ids = _scenario_evidence_ids(scenario)
    sequences = [row.sequence for row in spec.expected_steps]
    if sequences and sequences != list(range(1, len(sequences) + 1)):
        raise ValueError("Gold reviewed defense steps must be contiguous and ordered from 1")
    for step in spec.expected_steps:
        if step.target_entity_id not in entity_ids:
            raise ValueError(
                f"Gold review target is absent from scenario graph: {step.target_entity_id}"
            )
        missing_evidence = sorted(set(step.supporting_evidence_ids) - evidence_ids)
        if missing_evidence:
            raise ValueError(
                "Gold review cites supporting evidence absent from scenario graph: "
                + ", ".join(missing_evidence)
            )

    approved = spec.decision is CorrectionReviewDecision.APPROVE
    if packet.split is GoldReviewSplit.HOLDOUT:
        if spec.authorize_for_training:
            raise ValueError("Gold HOLDOUT packet cannot be authorized for training")
        if approved and not spec.authorize_for_evaluation:
            raise ValueError("approved Gold HOLDOUT packet requires evaluation authorization")
        training_authorization = False
        evaluation_authorization = approved and spec.authorize_for_evaluation
        promotion_eligible = False
    else:
        if approved and not spec.authorize_for_training:
            raise ValueError(
                "approved Gold TRAIN/VALIDATION packet requires explicit training authorization"
            )
        training_authorization = approved and spec.authorize_for_training
        evaluation_authorization = approved and spec.authorize_for_evaluation
        promotion_eligible = training_authorization and spec.outcome_verified

    payload: dict[str, object] = {
        "schema_version": "sentinel.gold-reviewed-packet.v1",
        "packet_id": packet.packet_id,
        "packet_sha256": packet.packet_sha256,
        "scenario_id": packet.scenario_id,
        "source_report_sha256": packet.source_report_sha256,
        "split": packet.split.value,
        "reviewer_id": spec.reviewer_id,
        "decision": spec.decision.value,
        "interpretation": spec.interpretation,
        "expected_steps": [row.model_dump(mode="json") for row in spec.expected_steps],
        "review_evidence_ids": list(dict.fromkeys(spec.review_evidence_ids)),
        "outcome_verified": approved and spec.outcome_verified,
        "training_authorization": training_authorization,
        "evaluation_authorization": evaluation_authorization,
        "promotion_eligible": promotion_eligible,
    }
    payload["review_sha256"] = _review_digest(payload)
    return GoldReviewedPacket.model_validate(payload)
