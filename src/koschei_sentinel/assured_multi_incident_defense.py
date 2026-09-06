from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.assured_active_defense import (
    ActiveDefenseAssurancePolicy,
    AssuredActiveDefensePlan,
    build_assured_active_defense_plan,
)
from koschei_sentinel.attack_progression import analyze_attack_progression
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.defense_authority import DefenseMode
from koschei_sentinel.models import StrictModel
from koschei_sentinel.perception_assurance import PerceptionAssuranceSummary
from koschei_sentinel.perception_graph_binding import PerceptionGraphReceipt
from koschei_sentinel.perception_source_registry import PerceptionSourceRegistry

_MODE_RANK = {
    DefenseMode.GUARD: 0,
    DefenseMode.COMBAT: 1,
    DefenseMode.SIEGE: 2,
}


class AssuredComponentDefensePlan(StrictModel):
    schema_version: Literal["sentinel.assured-component-defense-plan.v1"] = (
        "sentinel.assured-component-defense-plan.v1"
    )
    component_id: str
    entity_ids: list[str]
    critical_entity_ids: list[str]
    component_risk_score: float = Field(ge=0.0, le=1.0)
    priority_score: float = Field(ge=0.0, le=2.0)
    assured_plan: AssuredActiveDefensePlan

    @model_validator(mode="after")
    def component_boundary_is_preserved(self) -> AssuredComponentDefensePlan:
        entity_set = set(self.entity_ids)
        active = self.assured_plan.active_defense_plan
        if active.progression.primary_component_id != self.component_id:
            raise ValueError("assured component plan targets a different attack component")
        if not set(self.critical_entity_ids).issubset(entity_set):
            raise ValueError("component critical entities must remain inside the component")
        cut_entities = {
            row.entity_id
            for row in active.authorized_cut_points + active.withheld_cut_points
        }
        if not cut_entities.issubset(entity_set):
            raise ValueError("assured component cut points crossed an incident boundary")
        return self


class AssuredMultiIncidentDefensePlan(StrictModel):
    schema_version: Literal["sentinel.assured-multi-incident-defense-plan.v1"] = (
        "sentinel.assured-multi-incident-defense-plan.v1"
    )
    graph_id: str
    active_component_count: int = Field(ge=0)
    critical_component_count: int = Field(ge=0)
    high_impact_authorized_components: int = Field(ge=0)
    highest_effective_mode: DefenseMode
    component_plans: list[AssuredComponentDefensePlan]
    inactive_critical_entity_ids: list[str]
    rationale: list[str]

    @model_validator(mode="after")
    def components_are_unique(self) -> AssuredMultiIncidentDefensePlan:
        component_ids = [row.component_id for row in self.component_plans]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("assured multi-incident plan contains duplicate components")
        return self


def _priority(
    *,
    risk_score: float,
    critical: bool,
    mode: DefenseMode,
) -> float:
    return min(
        2.0,
        risk_score + (0.5 if critical else 0.0) + 0.1 * _MODE_RANK[mode],
    )


def build_assured_multi_incident_defense_plan(
    graph: CyberStateGraph,
    *,
    graph_receipt: PerceptionGraphReceipt,
    perception_assurance: PerceptionAssuranceSummary,
    registry: PerceptionSourceRegistry,
    critical_entity_ids: list[str] | None = None,
    policy: ActiveDefenseAssurancePolicy | None = None,
) -> AssuredMultiIncidentDefensePlan:
    critical = set(critical_entity_ids or [])
    known = {entity.entity_id for entity in graph.entities}
    unknown = sorted(critical - known)
    if unknown:
        raise ValueError(
            "critical_entity_ids reference unknown graph entities: " + ", ".join(unknown)
        )

    progression = analyze_attack_progression(graph)
    component_plans: list[AssuredComponentDefensePlan] = []
    active_critical_entities: set[str] = set()

    for component in progression.components:
        component_entities = set(component.entity_ids)
        component_critical = sorted(critical & component_entities)
        active_critical_entities.update(component_critical)
        assured = build_assured_active_defense_plan(
            graph,
            graph_receipt=graph_receipt,
            perception_assurance=perception_assurance,
            registry=registry,
            critical_entity_ids=component_critical,
            focus_entity_ids=[min(component.entity_ids)],
            policy=policy,
        )
        if assured.active_defense_plan.progression.primary_component_id != component.component_id:
            raise ValueError("assured component focus resolved to a different attack component")
        effective_mode = assured.assurance.effective_mode
        component_plans.append(
            AssuredComponentDefensePlan(
                component_id=component.component_id,
                entity_ids=component.entity_ids,
                critical_entity_ids=component_critical,
                component_risk_score=component.risk_score,
                priority_score=_priority(
                    risk_score=component.risk_score,
                    critical=bool(component_critical),
                    mode=effective_mode,
                ),
                assured_plan=assured,
            )
        )

    component_plans.sort(
        key=lambda row: (
            -row.priority_score,
            -_MODE_RANK[row.assured_plan.assurance.effective_mode],
            row.component_id,
        )
    )
    highest_mode = max(
        (row.assured_plan.assurance.effective_mode for row in component_plans),
        key=lambda mode: _MODE_RANK[mode],
        default=DefenseMode.GUARD,
    )
    high_impact = sum(
        row.assured_plan.assurance.high_impact_authorized for row in component_plans
    )
    critical_components = sum(bool(row.critical_entity_ids) for row in component_plans)
    inactive_critical = sorted(critical - active_critical_entities)

    rationale = [
        "every active attack component is independently evaluated by the perception assurance gate",
        "independence-domain authorization never transfers between disconnected incidents",
        "component cut points remain isolated before connector command generation",
        "critical components are prioritized without hiding non-critical active attacks",
    ]

    return AssuredMultiIncidentDefensePlan(
        graph_id=graph.graph_id,
        active_component_count=len(component_plans),
        critical_component_count=critical_components,
        high_impact_authorized_components=high_impact,
        highest_effective_mode=highest_mode,
        component_plans=component_plans,
        inactive_critical_entity_ids=inactive_critical,
        rationale=rationale,
    )
