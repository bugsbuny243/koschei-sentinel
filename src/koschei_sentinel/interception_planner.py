from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from koschei_sentinel.active_defense_planner import ActiveDefensePlan
from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.models import StrictModel


class InterceptionUrgency(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    HIGH = "HIGH"
    NORMAL = "NORMAL"


_ACTION_PRIORITY: dict[DefenseActionType, int] = {
    DefenseActionType.HOLD_TRANSACTION: 100,
    DefenseActionType.FREEZE_SIGNER: 95,
    DefenseActionType.PAUSE_PIPELINE: 90,
    DefenseActionType.REVOKE_CREDENTIAL: 85,
    DefenseActionType.TERMINATE_SESSION: 80,
    DefenseActionType.ISOLATE_ENDPOINT: 75,
    DefenseActionType.KILL_PROCESS: 70,
    DefenseActionType.QUARANTINE_WORKLOAD: 65,
    DefenseActionType.BLOCK_IOC: 60,
    DefenseActionType.COLLECT_EVIDENCE: 30,
    DefenseActionType.OBSERVE: 10,
    DefenseActionType.ENABLE_EMERGENCY_POLICY: 5,
}

_IMPACT_EDGE_ACTIONS = {
    DefenseActionType.HOLD_TRANSACTION,
    DefenseActionType.FREEZE_SIGNER,
}


class InterceptionStep(StrictModel):
    step_id: str
    sequence: int = Field(ge=1)
    action: DefenseActionType
    target_entity_id: str
    urgency: InterceptionUrgency
    expected_effect_score: float = Field(ge=0.0, le=1.0)
    supporting_relation_ids: list[str]
    verification_required: bool = True
    stop_if_verified: bool = False
    objective: str


class InterceptionPlan(StrictModel):
    schema_version: Literal["sentinel.interception-plan.v1"] = (
        "sentinel.interception-plan.v1"
    )
    graph_id: str
    defense_mode: DefenseMode
    steps: list[InterceptionStep]
    predicted_impact_present: bool
    rationale: list[str]


def _urgency(effect: float, action: DefenseActionType, predicted_impact: bool) -> InterceptionUrgency:
    if action in _IMPACT_EDGE_ACTIONS and predicted_impact:
        return InterceptionUrgency.IMMEDIATE
    if effect >= 0.8:
        return InterceptionUrgency.IMMEDIATE
    if effect >= 0.55:
        return InterceptionUrgency.HIGH
    return InterceptionUrgency.NORMAL


def _objective(action: DefenseActionType) -> str:
    return {
        DefenseActionType.HOLD_TRANSACTION: "prevent a pending hostile state transition",
        DefenseActionType.FREEZE_SIGNER: "remove signing authority from the active hostile path",
        DefenseActionType.PAUSE_PIPELINE: "stop compromised build or release propagation",
        DefenseActionType.REVOKE_CREDENTIAL: "invalidate attacker-held authorization material",
        DefenseActionType.TERMINATE_SESSION: "cut the active authenticated attacker session",
        DefenseActionType.ISOLATE_ENDPOINT: "break lateral movement and command paths from the endpoint",
        DefenseActionType.KILL_PROCESS: "stop the malicious execution primitive",
        DefenseActionType.QUARANTINE_WORKLOAD: "remove the compromised workload from reachable infrastructure",
        DefenseActionType.BLOCK_IOC: "deny the observed hostile network indicator",
        DefenseActionType.COLLECT_EVIDENCE: "increase confidence before higher-impact containment",
        DefenseActionType.OBSERVE: "maintain observation while evidence remains insufficient",
        DefenseActionType.ENABLE_EMERGENCY_POLICY: "activate pre-defined emergency defensive controls",
    }.get(action, "interrupt hostile progression")


def _ranking_key(row: object, predicted_impact: bool) -> tuple[float, float, float, str]:
    action = row.action
    impact_edge = 1.0 if predicted_impact and action in _IMPACT_EDGE_ACTIONS else 0.0
    return (
        -impact_edge,
        -row.effect_score,
        -float(_ACTION_PRIORITY.get(action, 0)),
        row.entity_id,
    )


def build_interception_plan(plan: ActiveDefensePlan) -> InterceptionPlan:
    predicted_impact = any(
        transition.to_stage.value == "IMPACT"
        for transition in plan.progression.predicted_transitions
    )

    ranked = sorted(
        plan.authorized_cut_points,
        key=lambda row: _ranking_key(row, predicted_impact),
    )

    steps: list[InterceptionStep] = []
    for index, cut in enumerate(ranked, 1):
        steps.append(
            InterceptionStep(
                step_id=f"intercept:{plan.graph_id}:{index}",
                sequence=index,
                action=cut.action,
                target_entity_id=cut.entity_id,
                urgency=_urgency(cut.effect_score, cut.action, predicted_impact),
                expected_effect_score=cut.effect_score,
                supporting_relation_ids=cut.supporting_relation_ids,
                verification_required=True,
                stop_if_verified=cut.effect_score >= 0.9,
                objective=_objective(cut.action),
            )
        )

    rationale = [
        "only cut points already authorized by the selected defense mode are eligible",
        "when impact is predicted, signer and pending-transaction controls are sequenced before broader graph cuts",
        "otherwise steps are ranked by graph-level interruption effect with action priority as a tie-breaker",
        "every containment step requires post-action verification before the incident is considered contained",
    ]
    if predicted_impact:
        rationale.append("predicted impact raises urgency for signer and transaction controls")
    if not steps:
        rationale.append("no higher-impact containment step is authorized in the current defense mode")

    return InterceptionPlan(
        graph_id=plan.graph_id,
        defense_mode=plan.decision.mode,
        steps=steps,
        predicted_impact_present=predicted_impact,
        rationale=rationale,
    )
