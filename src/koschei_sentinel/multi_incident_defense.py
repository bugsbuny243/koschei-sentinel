from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.active_defense_planner import ActiveDefensePlan, build_active_defense_plan
from koschei_sentinel.attack_progression import AttackComponentReport, analyze_attack_progression
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.defense_authority import DefenseMode
from koschei_sentinel.models import StrictModel

_MODE_RANK = {
    DefenseMode.GUARD: 0,
    DefenseMode.COMBAT: 1,
    DefenseMode.SIEGE: 2,
}


class ComponentDefensePlan(StrictModel):
    schema_version: Literal["sentinel.component-defense-plan.v1"] = (
        "sentinel.component-defense-plan.v1"
    )
    component_id: str
    entity_ids: list[str]
    critical_entity_ids: list[str]
    component_risk_score: float = Field(ge=0.0, le=1.0)
    priority_score: float = Field(ge=0.0, le=2.0)
    defense_plan: ActiveDefensePlan

    @model_validator(mode="after")
    def component_boundary_is_preserved(self) -> ComponentDefensePlan:
        entity_set = set(self.entity_ids)
        if not set(self.critical_entity_ids).issubset(entity_set):
            raise ValueError("component critical entities must remain inside the component")
        if self.defense_plan.progression.primary_component_id != self.component_id:
            raise ValueError("component defense plan targets a different progression component")
        cut_entities = {
            row.entity_id
            for row in (
                self.defense_plan.authorized_cut_points
                + self.defense_plan.withheld_cut_points
            )
        }
        if not cut_entities.issubset(entity_set):
            raise ValueError("component defense cut points crossed an incident boundary")
        return self


class MultiIncidentDefensePlan(StrictModel):
    schema_version: Literal["sentinel.multi-incident-defense-plan.v1"] = (
        "sentinel.multi-incident-defense-plan.v1"
    )
    graph_id: str
    active_component_count: int = Field(ge=0)
    critical_component_count: int = Field(ge=0)
    highest_mode: DefenseMode
    component_plans: list[ComponentDefensePlan]
    inactive_critical_entity_ids: list[str]
    rationale: list[str]

    @model_validator(mode="after")
    def component_ids_are_unique(self) -> MultiIncidentDefensePlan:
        component_ids = [row.component_id for row in self.component_plans]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("multi-incident defense plan contains duplicate components")
        return self


def _component_priority(component: AttackComponentReport, has_critical: bool, mode: DefenseMode) -> float:
    critical_bonus = 0.5 if has_critical else 0.0
    mode_bonus = 0.1 * _MODE_RANK[mode]
    return min(2.0, component.risk_score + critical_bonus + mode_bonus)


def build_multi_incident_defense_plan(
    graph: CyberStateGraph,
    *,
    critical_entity_ids: list[str] | None = None,
) -> MultiIncidentDefensePlan:
    critical = set(critical_entity_ids or [])
    known = {entity.entity_id for entity in graph.entities}
    unknown = sorted(critical - known)
    if unknown:
        raise ValueError(
            "critical_entity_ids reference unknown graph entities: " + ", ".join(unknown)
        )

    progression = analyze_attack_progression(graph)
    component_plans: list[ComponentDefensePlan] = []
    active_critical_entities: set[str] = set()

    for component in progression.components:
        component_entities = set(component.entity_ids)
        component_critical = sorted(critical & component_entities)
        active_critical_entities.update(component_critical)
        focus = [min(component.entity_ids)]
        plan = build_active_defense_plan(
            graph,
            critical_entity_ids=component_critical,
            focus_entity_ids=focus,
        )
        if plan.progression.primary_component_id != component.component_id:
            raise ValueError("component focus resolved to a different attack component")
        component_plans.append(
            ComponentDefensePlan(
                component_id=component.component_id,
                entity_ids=component.entity_ids,
                critical_entity_ids=component_critical,
                component_risk_score=component.risk_score,
                priority_score=_component_priority(
                    component,
                    bool(component_critical),
                    plan.decision.mode,
                ),
                defense_plan=plan,
            )
        )

    component_plans.sort(
        key=lambda row: (
            -row.priority_score,
            -_MODE_RANK[row.defense_plan.decision.mode],
            row.component_id,
        )
    )
    highest_mode = max(
        (row.defense_plan.decision.mode for row in component_plans),
        key=lambda mode: _MODE_RANK[mode],
        default=DefenseMode.GUARD,
    )
    inactive_critical = sorted(critical - active_critical_entities)
    critical_components = sum(bool(row.critical_entity_ids) for row in component_plans)

    rationale = [
        "each connected active attack component receives an independent defense plan",
        "cut points are forbidden from crossing component boundaries",
        "critical-asset components receive a deterministic scheduling priority bonus",
        "independent incidents remain visible even when a different component is higher risk",
    ]
    if inactive_critical:
        rationale.append(
            "declared critical assets without an active component remain protected but require no incident plan"
        )

    return MultiIncidentDefensePlan(
        graph_id=graph.graph_id,
        active_component_count=len(component_plans),
        critical_component_count=critical_components,
        highest_mode=highest_mode,
        component_plans=component_plans,
        inactive_critical_entity_ids=inactive_critical,
        rationale=rationale,
    )
