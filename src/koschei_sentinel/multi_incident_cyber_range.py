from __future__ import annotations

from enum import StrEnum
from statistics import mean
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.attack_world_lines import (
    AttackWorldLineTimeline,
    WorldLineTransitionType,
    build_attack_world_line_timeline,
)
from koschei_sentinel.cyber_range import SimulatedActionOutcome
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode
from koschei_sentinel.interception_execution import (
    authorize_next_step,
    record_step_execution,
    start_interception_execution,
    verify_step_outcome,
)
from koschei_sentinel.interception_planner import build_interception_plan
from koschei_sentinel.models import StrictModel
from koschei_sentinel.multi_incident_defense import build_multi_incident_defense_plan


class MultiIncidentScenarioTruth(StrEnum):
    MALICIOUS = "MALICIOUS"
    MIXED = "MIXED"
    BENIGN = "BENIGN"


class ExpectedWorldLineTransition(StrictModel):
    to_tick: int = Field(ge=1)
    transition_type: WorldLineTransitionType
    protected_anchor_id: str | None = None


class MultiIncidentCyberRangeScenario(StrictModel):
    schema_version: Literal["sentinel.multi-incident-cyber-range-scenario.v1"] = (
        "sentinel.multi-incident-cyber-range-scenario.v1"
    )
    scenario_id: str = Field(min_length=3, max_length=256)
    truth: MultiIncidentScenarioTruth
    graph_snapshots: list[CyberStateGraph] = Field(min_length=1, max_length=128)
    critical_entity_ids: list[str] = Field(default_factory=list)
    expected_active_component_counts: list[int] = Field(default_factory=list)
    expected_world_line_transitions: list[ExpectedWorldLineTransition] = Field(default_factory=list)
    action_outcomes: list[SimulatedActionOutcome] = Field(default_factory=list)

    @model_validator(mode="after")
    def scenario_is_coherent(self) -> MultiIncidentCyberRangeScenario:
        graph_ids = {graph.graph_id for graph in self.graph_snapshots}
        if len(graph_ids) != 1:
            raise ValueError("all graph snapshots in a multi-incident scenario must share graph_id")
        if self.expected_active_component_counts and len(self.expected_active_component_counts) != len(
            self.graph_snapshots
        ):
            raise ValueError(
                "expected_active_component_counts must be empty or match graph snapshot count"
            )
        for expected in self.expected_world_line_transitions:
            if expected.to_tick >= len(self.graph_snapshots):
                raise ValueError("expected world-line transition references an unavailable tick")
        return self


class ComponentRangeTick(StrictModel):
    component_id: str
    entity_ids: list[str]
    critical_entity_ids: list[str]
    defense_mode: DefenseMode
    authorized_actions: list[DefenseActionType]
    attempted_actions: list[DefenseActionType]
    succeeded_actions: list[DefenseActionType]
    contained: bool
    cut_point_leakage: bool


class MultiIncidentRangeTick(StrictModel):
    tick: int = Field(ge=0)
    graph_id: str
    active_component_count: int = Field(ge=0)
    component_plans: list[ComponentRangeTick]
    cut_point_leakage_count: int = Field(ge=0)


class MultiIncidentCyberRangeReport(StrictModel):
    schema_version: Literal["sentinel.multi-incident-cyber-range-report.v1"] = (
        "sentinel.multi-incident-cyber-range-report.v1"
    )
    scenario_id: str
    truth: MultiIncidentScenarioTruth
    ticks: list[MultiIncidentRangeTick]
    world_lines: AttackWorldLineTimeline
    expected_component_ticks: int = Field(ge=0)
    component_count_matches: int = Field(ge=0)
    component_count_accuracy: float = Field(ge=0.0, le=1.0)
    total_component_instances: int = Field(ge=0)
    contained_component_instances: int = Field(ge=0)
    component_containment_rate: float = Field(ge=0.0, le=1.0)
    mean_containment_step: float | None = Field(default=None, ge=0.0)
    world_line_count: int = Field(ge=0)
    contained_world_lines: int = Field(ge=0)
    world_line_containment_rate: float = Field(ge=0.0, le=1.0)
    mean_world_line_containment_tick_latency: float | None = Field(default=None, ge=0.0)
    uncontained_world_line_ids: list[str]
    cut_point_leakage_count: int = Field(ge=0)
    expected_world_line_transitions: int = Field(ge=0)
    matched_world_line_transitions: int = Field(ge=0)
    world_line_transition_accuracy: float = Field(ge=0.0, le=1.0)
    missed_world_line_transitions: list[str]
    passed: bool
    violations: list[str]


def _outcome_map(
    scenario: MultiIncidentCyberRangeScenario,
) -> dict[tuple[DefenseActionType, str], bool]:
    return {
        (row.action, row.target_entity_id): row.succeeds
        for row in scenario.action_outcomes
    }


def _step_evidence_ids(graph: CyberStateGraph, relation_ids: list[str]) -> list[str]:
    supporting = set(relation_ids)
    evidence: list[str] = []
    for relation in graph.relations:
        if relation.relation_id in supporting:
            evidence.extend(item.evidence_id for item in relation.evidence)
    return list(dict.fromkeys(evidence))


def _simulate_component(
    *,
    scenario: MultiIncidentCyberRangeScenario,
    graph: CyberStateGraph,
    defense_plan: object,
) -> tuple[list[DefenseActionType], list[DefenseActionType], bool, int | None]:
    interception = build_interception_plan(defense_plan)
    execution = start_interception_execution(interception)
    attempted: list[DefenseActionType] = []
    succeeded: list[DefenseActionType] = []
    first_successful_stop_step: int | None = None
    outcomes = _outcome_map(scenario)

    while not execution.complete:
        pending = next(
            (row for row in execution.steps if row.status.value == "PENDING"),
            None,
        )
        if pending is None:
            break
        step = next(row for row in interception.steps if row.step_id == pending.step_id)
        evidence_ids = _step_evidence_ids(graph, step.supporting_relation_ids)
        if not evidence_ids:
            break
        execution = authorize_next_step(
            execution,
            interception,
            precondition_evidence_ids=evidence_ids,
        )
        attempted.append(step.action)
        execution = record_step_execution(
            execution,
            interception,
            step_id=step.step_id,
            execution_receipt_ids=[
                f"multi-range:receipt:{scenario.scenario_id}:{step.step_id}"
            ],
        )
        ok = outcomes.get((step.action, step.target_entity_id), True)
        execution = verify_step_outcome(
            execution,
            interception,
            step_id=step.step_id,
            succeeded=ok,
            outcome_evidence_ids=[
                f"multi-range:outcome:{scenario.scenario_id}:{step.step_id}:{int(ok)}"
            ],
        )
        if ok:
            succeeded.append(step.action)
            if step.stop_if_verified and first_successful_stop_step is None:
                first_successful_stop_step = step.sequence

    return attempted, succeeded, execution.contained, first_successful_stop_step


def _matches_expected_transition(
    timeline: AttackWorldLineTimeline,
    expected: ExpectedWorldLineTransition,
) -> bool:
    for transition in timeline.transitions:
        if transition.to_tick != expected.to_tick:
            continue
        if transition.transition_type is not expected.transition_type:
            continue
        if expected.protected_anchor_id is not None:
            if expected.protected_anchor_id not in transition.shared_protected_anchor_ids:
                continue
        return True
    return False


def _world_line_containment_metrics(
    timeline: AttackWorldLineTimeline,
    containment_by_component_tick: dict[tuple[int, str], bool],
) -> tuple[int, int, float, float | None, list[str]]:
    observations_by_line: dict[str, list[object]] = {}
    for observation in timeline.observations:
        observations_by_line.setdefault(observation.world_line_id, []).append(observation)

    latencies: list[float] = []
    uncontained: list[str] = []
    for line_id, observations in sorted(observations_by_line.items()):
        ordered = sorted(observations, key=lambda row: (row.tick, row.component_id))
        first_tick = ordered[0].tick
        contained_tick: int | None = None
        for observation in ordered:
            if containment_by_component_tick.get(
                (observation.tick, observation.component_id),
                False,
            ):
                contained_tick = observation.tick
                break
        if contained_tick is None:
            uncontained.append(line_id)
        else:
            latencies.append(float(contained_tick - first_tick))

    total = len(observations_by_line)
    contained = total - len(uncontained)
    rate = 1.0 if total == 0 else contained / total
    return total, contained, rate, (mean(latencies) if latencies else None), uncontained


def run_multi_incident_cyber_range(
    scenario: MultiIncidentCyberRangeScenario,
) -> MultiIncidentCyberRangeReport:
    tick_reports: list[MultiIncidentRangeTick] = []
    containment_steps: list[float] = []
    containment_by_component_tick: dict[tuple[int, str], bool] = {}
    total_components = 0
    contained_components = 0
    leakage_total = 0

    for tick, graph in enumerate(scenario.graph_snapshots):
        multi = build_multi_incident_defense_plan(
            graph,
            critical_entity_ids=scenario.critical_entity_ids,
        )
        rows: list[ComponentRangeTick] = []
        for component in multi.component_plans:
            entity_set = set(component.entity_ids)
            cut_points = (
                component.defense_plan.authorized_cut_points
                + component.defense_plan.withheld_cut_points
            )
            leakage = any(row.entity_id not in entity_set for row in cut_points)
            attempted, succeeded, contained, containment_step = _simulate_component(
                scenario=scenario,
                graph=graph,
                defense_plan=component.defense_plan,
            )
            containment_by_component_tick[(tick, component.component_id)] = contained
            total_components += 1
            if contained:
                contained_components += 1
            if containment_step is not None:
                containment_steps.append(float(containment_step))
            leakage_total += int(leakage)
            rows.append(
                ComponentRangeTick(
                    component_id=component.component_id,
                    entity_ids=component.entity_ids,
                    critical_entity_ids=component.critical_entity_ids,
                    defense_mode=component.defense_plan.decision.mode,
                    authorized_actions=[
                        row.action for row in component.defense_plan.authorized_cut_points
                    ],
                    attempted_actions=attempted,
                    succeeded_actions=succeeded,
                    contained=contained,
                    cut_point_leakage=leakage,
                )
            )
        tick_reports.append(
            MultiIncidentRangeTick(
                tick=tick,
                graph_id=graph.graph_id,
                active_component_count=multi.active_component_count,
                component_plans=rows,
                cut_point_leakage_count=sum(row.cut_point_leakage for row in rows),
            )
        )

    world_lines = build_attack_world_line_timeline(
        scenario.graph_snapshots,
        stream_id=f"multi-range:{scenario.scenario_id}",
        protected_anchor_entity_ids=scenario.critical_entity_ids,
    )
    (
        world_line_count,
        contained_world_lines,
        world_line_containment_rate,
        mean_world_line_latency,
        uncontained_world_lines,
    ) = _world_line_containment_metrics(world_lines, containment_by_component_tick)

    expected_component_ticks = len(scenario.expected_active_component_counts)
    component_matches = sum(
        tick.active_component_count == expected
        for tick, expected in zip(
            tick_reports,
            scenario.expected_active_component_counts,
            strict=False,
        )
    )
    component_accuracy = (
        1.0
        if expected_component_ticks == 0
        else component_matches / expected_component_ticks
    )

    expected_transitions = len(scenario.expected_world_line_transitions)
    missed: list[str] = []
    matched_transitions = 0
    for expected in scenario.expected_world_line_transitions:
        if _matches_expected_transition(world_lines, expected):
            matched_transitions += 1
        else:
            missed.append(
                f"tick={expected.to_tick}:{expected.transition_type.value}:"
                f"anchor={expected.protected_anchor_id or '-'}"
            )
    transition_accuracy = (
        1.0 if expected_transitions == 0 else matched_transitions / expected_transitions
    )

    containment_rate = 1.0 if total_components == 0 else contained_components / total_components
    violations: list[str] = []
    if component_accuracy < 1.0:
        violations.append("one or more expected active component counts were incorrect")
    if leakage_total:
        violations.append("one or more defensive cut points crossed an incident component boundary")
    if transition_accuracy < 1.0:
        violations.append("one or more expected world-line transitions were missed")
    if scenario.truth is MultiIncidentScenarioTruth.MALICIOUS:
        if total_components and containment_rate < 1.0:
            violations.append("one or more malicious component instances were not contained")
        if world_line_count and world_line_containment_rate < 1.0:
            violations.append("one or more malicious attack world-lines were not contained")

    return MultiIncidentCyberRangeReport(
        scenario_id=scenario.scenario_id,
        truth=scenario.truth,
        ticks=tick_reports,
        world_lines=world_lines,
        expected_component_ticks=expected_component_ticks,
        component_count_matches=component_matches,
        component_count_accuracy=component_accuracy,
        total_component_instances=total_components,
        contained_component_instances=contained_components,
        component_containment_rate=containment_rate,
        mean_containment_step=(mean(containment_steps) if containment_steps else None),
        world_line_count=world_line_count,
        contained_world_lines=contained_world_lines,
        world_line_containment_rate=world_line_containment_rate,
        mean_world_line_containment_tick_latency=mean_world_line_latency,
        uncontained_world_line_ids=uncontained_world_lines,
        cut_point_leakage_count=leakage_total,
        expected_world_line_transitions=expected_transitions,
        matched_world_line_transitions=matched_transitions,
        world_line_transition_accuracy=transition_accuracy,
        missed_world_line_transitions=missed,
        passed=not violations,
        violations=violations,
    )
