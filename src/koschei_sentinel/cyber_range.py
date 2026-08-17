from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.active_defense_planner import build_active_defense_plan
from koschei_sentinel.adaptive_defense import ReassessmentDisposition, reassess_after_interception
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.interception_execution import (
    InterceptionExecution,
    authorize_next_step,
    record_step_execution,
    start_interception_execution,
    verify_step_outcome,
)
from koschei_sentinel.interception_planner import InterceptionPlan, build_interception_plan
from koschei_sentinel.models import StrictModel


class ScenarioTruth(StrEnum):
    MALICIOUS = "MALICIOUS"
    BENIGN = "BENIGN"


class SimulatedActionOutcome(StrictModel):
    action: DefenseActionType
    target_entity_id: str
    succeeds: bool


class CyberRangeScenario(StrictModel):
    schema_version: Literal["sentinel.cyber-range-scenario.v1"] = (
        "sentinel.cyber-range-scenario.v1"
    )
    scenario_id: str = Field(min_length=3, max_length=256)
    truth: ScenarioTruth
    graph_snapshots: list[CyberStateGraph] = Field(min_length=1, max_length=128)
    critical_entity_ids: list[str] = Field(default_factory=list)
    expected_reroute_ticks: list[int] = Field(default_factory=list)
    action_outcomes: list[SimulatedActionOutcome] = Field(default_factory=list)

    @model_validator(mode="after")
    def scenario_is_coherent(self) -> "CyberRangeScenario":
        graph_ids = {graph.graph_id for graph in self.graph_snapshots}
        if len(graph_ids) != 1:
            raise ValueError("all graph snapshots in a cyber-range scenario must share graph_id")
        for tick in self.expected_reroute_ticks:
            if tick <= 0 or tick >= len(self.graph_snapshots):
                raise ValueError("expected_reroute_ticks must reference a non-initial graph snapshot")
        return self


class CyberRangeTick(StrictModel):
    tick: int = Field(ge=0)
    graph_id: str
    defense_mode: DefenseMode
    current_stage: str | None
    attack_confidence: float = Field(ge=0.0, le=1.0)
    authorized_actions: list[DefenseActionType]
    attempted_actions: list[DefenseActionType]
    succeeded_actions: list[DefenseActionType]
    contained: bool
    reassessment_disposition: ReassessmentDisposition | None = None
    reroute_expected: bool = False
    reroute_detected: bool = False


class CyberRangeReport(StrictModel):
    schema_version: Literal["sentinel.cyber-range-report.v1"] = (
        "sentinel.cyber-range-report.v1"
    )
    scenario_id: str
    truth: ScenarioTruth
    ticks: list[CyberRangeTick]
    containment_tick: int | None
    reroutes_expected: int = Field(ge=0)
    reroutes_detected: int = Field(ge=0)
    reroute_detection_rate: float = Field(ge=0.0, le=1.0)
    false_positive_high_impact_actions: int = Field(ge=0)
    high_impact_actions_attempted: int = Field(ge=0)
    passed: bool
    violations: list[str]


_HIGH_IMPACT_ACTIONS = {
    DefenseActionType.REVOKE_CREDENTIAL,
    DefenseActionType.TERMINATE_SESSION,
    DefenseActionType.KILL_PROCESS,
    DefenseActionType.ISOLATE_ENDPOINT,
    DefenseActionType.QUARANTINE_WORKLOAD,
    DefenseActionType.PAUSE_PIPELINE,
    DefenseActionType.FREEZE_SIGNER,
    DefenseActionType.HOLD_TRANSACTION,
    DefenseActionType.ENABLE_EMERGENCY_POLICY,
}


def _outcome_map(scenario: CyberRangeScenario) -> dict[tuple[DefenseActionType, str], bool]:
    return {
        (row.action, row.target_entity_id): row.succeeds
        for row in scenario.action_outcomes
    }


def _precondition_evidence_ids(graph: CyberStateGraph, plan_step: object) -> list[str]:
    supporting = set(plan_step.supporting_relation_ids)
    evidence: list[str] = []
    for relation in graph.relations:
        if relation.relation_id in supporting:
            evidence.extend(item.evidence_id for item in relation.evidence)
    # Inferred cut points can exist with sparse evidence. The simulator must not invent
    # execution authority; no evidence means that step cannot be authorized.
    return list(dict.fromkeys(evidence))


def _simulate_interception(
    *,
    scenario: CyberRangeScenario,
    graph: CyberStateGraph,
    plan: InterceptionPlan,
    tick: int,
) -> tuple[InterceptionExecution, list[DefenseActionType], list[DefenseActionType]]:
    execution = start_interception_execution(plan)
    attempted: list[DefenseActionType] = []
    succeeded: list[DefenseActionType] = []
    outcomes = _outcome_map(scenario)

    while not execution.complete:
        pending = next(
            (row for row in execution.steps if row.status.value == "PENDING"),
            None,
        )
        if pending is None:
            break
        plan_step = next(row for row in plan.steps if row.step_id == pending.step_id)
        evidence_ids = _precondition_evidence_ids(graph, plan_step)
        if not evidence_ids:
            break

        execution = authorize_next_step(
            execution,
            plan,
            precondition_evidence_ids=evidence_ids,
        )
        attempted.append(plan_step.action)
        execution = record_step_execution(
            execution,
            plan,
            step_id=plan_step.step_id,
            execution_receipt_ids=[f"sim:receipt:{scenario.scenario_id}:{tick}:{plan_step.sequence}"],
        )
        ok = outcomes.get((plan_step.action, plan_step.target_entity_id), True)
        execution = verify_step_outcome(
            execution,
            plan,
            step_id=plan_step.step_id,
            succeeded=ok,
            outcome_evidence_ids=[
                f"sim:outcome:{scenario.scenario_id}:{tick}:{plan_step.sequence}:{int(ok)}"
            ],
        )
        if ok:
            succeeded.append(plan_step.action)

    return execution, attempted, succeeded


def run_cyber_range_scenario(scenario: CyberRangeScenario) -> CyberRangeReport:
    ticks: list[CyberRangeTick] = []
    containment_tick: int | None = None
    prior_graph: CyberStateGraph | None = None
    prior_execution: InterceptionExecution | None = None
    reroutes_detected = 0
    high_impact_attempted = 0
    false_positive_high_impact = 0
    violations: list[str] = []

    reroute_ticks = set(scenario.expected_reroute_ticks)

    for tick, graph in enumerate(scenario.graph_snapshots):
        active = build_active_defense_plan(
            graph,
            critical_entity_ids=scenario.critical_entity_ids,
        )
        interception = build_interception_plan(active)
        execution, attempted, succeeded = _simulate_interception(
            scenario=scenario,
            graph=graph,
            plan=interception,
            tick=tick,
        )

        high_impact = sum(action in _HIGH_IMPACT_ACTIONS for action in attempted)
        high_impact_attempted += high_impact
        if scenario.truth is ScenarioTruth.BENIGN:
            false_positive_high_impact += high_impact

        disposition: ReassessmentDisposition | None = None
        reroute_expected = tick in reroute_ticks
        reroute_detected = False
        if prior_graph is not None and prior_execution is not None:
            reassessment = reassess_after_interception(
                previous_graph=prior_graph,
                updated_graph=graph,
                previous_execution=prior_execution,
                critical_entity_ids=scenario.critical_entity_ids,
            )
            disposition = reassessment.disposition
            if reroute_expected and disposition in {
                ReassessmentDisposition.CONTINUE_CONTAINMENT,
                ReassessmentDisposition.HUNT_ALTERNATE_PATHS,
            }:
                reroute_detected = True
                reroutes_detected += 1

        if execution.contained and containment_tick is None:
            containment_tick = tick

        ticks.append(
            CyberRangeTick(
                tick=tick,
                graph_id=graph.graph_id,
                defense_mode=active.decision.mode,
                current_stage=(
                    active.progression.current_stage.value
                    if active.progression.current_stage is not None
                    else None
                ),
                attack_confidence=active.assessment.attack_confidence,
                authorized_actions=[row.action for row in active.authorized_cut_points],
                attempted_actions=attempted,
                succeeded_actions=succeeded,
                contained=execution.contained,
                reassessment_disposition=disposition,
                reroute_expected=reroute_expected,
                reroute_detected=reroute_detected,
            )
        )
        prior_graph = graph
        prior_execution = execution

    expected = len(reroute_ticks)
    reroute_rate = 1.0 if expected == 0 else reroutes_detected / expected

    if scenario.truth is ScenarioTruth.BENIGN and false_positive_high_impact:
        violations.append("benign scenario triggered high-impact containment")
    if scenario.truth is ScenarioTruth.MALICIOUS and containment_tick is None:
        violations.append("malicious scenario was not contained")
    if expected and reroutes_detected != expected:
        violations.append("one or more expected attacker reroutes were not detected")

    return CyberRangeReport(
        scenario_id=scenario.scenario_id,
        truth=scenario.truth,
        ticks=ticks,
        containment_tick=containment_tick,
        reroutes_expected=expected,
        reroutes_detected=reroutes_detected,
        reroute_detection_rate=reroute_rate,
        false_positive_high_impact_actions=false_positive_high_impact,
        high_impact_actions_attempted=high_impact_attempted,
        passed=not violations,
        violations=violations,
    )
