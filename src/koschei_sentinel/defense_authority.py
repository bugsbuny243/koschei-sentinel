from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class DefenseMode(StrEnum):
    GUARD = "GUARD"
    COMBAT = "COMBAT"
    SIEGE = "SIEGE"


class DefenseActionType(StrEnum):
    OBSERVE = "OBSERVE"
    COLLECT_EVIDENCE = "COLLECT_EVIDENCE"
    BLOCK_IOC = "BLOCK_IOC"
    REVOKE_CREDENTIAL = "REVOKE_CREDENTIAL"
    TERMINATE_SESSION = "TERMINATE_SESSION"
    KILL_PROCESS = "KILL_PROCESS"
    ISOLATE_ENDPOINT = "ISOLATE_ENDPOINT"
    QUARANTINE_WORKLOAD = "QUARANTINE_WORKLOAD"
    PAUSE_PIPELINE = "PAUSE_PIPELINE"
    FREEZE_SIGNER = "FREEZE_SIGNER"
    HOLD_TRANSACTION = "HOLD_TRANSACTION"
    ENABLE_EMERGENCY_POLICY = "ENABLE_EMERGENCY_POLICY"


class AttackAssessment(StrictModel):
    schema_version: Literal["sentinel.attack-assessment.v1"] = "sentinel.attack-assessment.v1"
    assessment_id: str = Field(min_length=3, max_length=256)
    attack_confidence: float = Field(ge=0.0, le=1.0)
    corroborating_evidence_count: int = Field(ge=0)
    critical_asset_at_risk: bool = False
    active_progression: bool = False
    blast_radius_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_gap: bool = False


class DefenseDecision(StrictModel):
    schema_version: Literal["sentinel.defense-decision.v1"] = "sentinel.defense-decision.v1"
    mode: DefenseMode
    permitted_actions: list[DefenseActionType]
    rationale: list[str]


_GUARD_ACTIONS = [
    DefenseActionType.OBSERVE,
    DefenseActionType.COLLECT_EVIDENCE,
    DefenseActionType.BLOCK_IOC,
]

_COMBAT_ACTIONS = _GUARD_ACTIONS + [
    DefenseActionType.REVOKE_CREDENTIAL,
    DefenseActionType.TERMINATE_SESSION,
    DefenseActionType.KILL_PROCESS,
    DefenseActionType.ISOLATE_ENDPOINT,
    DefenseActionType.QUARANTINE_WORKLOAD,
    DefenseActionType.PAUSE_PIPELINE,
    DefenseActionType.FREEZE_SIGNER,
    DefenseActionType.HOLD_TRANSACTION,
]

_SIEGE_ACTIONS = _COMBAT_ACTIONS + [DefenseActionType.ENABLE_EMERGENCY_POLICY]


def decide_defense_mode(assessment: AttackAssessment) -> DefenseDecision:
    rationale: list[str] = []

    if (
        assessment.attack_confidence >= 0.95
        and assessment.active_progression
        and assessment.critical_asset_at_risk
        and assessment.blast_radius_score >= 0.7
        and assessment.corroborating_evidence_count >= 2
    ):
        rationale.append("high-confidence active attack threatens critical assets")
        rationale.append("broad blast radius justifies emergency defensive containment")
        return DefenseDecision(mode=DefenseMode.SIEGE, permitted_actions=_SIEGE_ACTIONS, rationale=rationale)

    if (
        assessment.attack_confidence >= 0.8
        and assessment.corroborating_evidence_count >= 2
        and (assessment.active_progression or assessment.critical_asset_at_risk)
    ):
        rationale.append("corroborated attack confidence meets active-defense threshold")
        return DefenseDecision(mode=DefenseMode.COMBAT, permitted_actions=_COMBAT_ACTIONS, rationale=rationale)

    rationale.append("attack evidence remains below active-containment threshold")
    if assessment.evidence_gap:
        rationale.append("evidence gap requires additional collection")
    return DefenseDecision(mode=DefenseMode.GUARD, permitted_actions=_GUARD_ACTIONS, rationale=rationale)


class DefenseExecutionRecord(StrictModel):
    schema_version: Literal["sentinel.defense-execution.v1"] = "sentinel.defense-execution.v1"
    action: DefenseActionType
    target: str = Field(min_length=2, max_length=1024)
    mode: DefenseMode
    authorized: bool
    precondition_evidence_ids: list[str] = Field(default_factory=list)
    outcome_verified: bool = False
    outcome_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def execution_is_auditable(self) -> "DefenseExecutionRecord":
        if self.authorized and not self.precondition_evidence_ids:
            raise ValueError("authorized defense execution requires precondition evidence")
        if self.outcome_verified and not self.outcome_evidence_ids:
            raise ValueError("verified outcome requires outcome evidence")
        return self
