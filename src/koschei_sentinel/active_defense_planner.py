from __future__ import annotations

from typing import Literal

from pydantic import Field

from koschei_sentinel.attack_progression import (
    AttackProgressionReport,
    DefensiveCutPoint,
    analyze_attack_progression,
)
from koschei_sentinel.cyber_state_graph import CyberStateGraph, EvidenceStatus
from koschei_sentinel.defense_authority import (
    AttackAssessment,
    DefenseDecision,
    decide_defense_mode,
)
from koschei_sentinel.models import StrictModel


class ActiveDefensePlan(StrictModel):
    schema_version: Literal["sentinel.active-defense-plan.v1"] = (
        "sentinel.active-defense-plan.v1"
    )
    graph_id: str
    assessment: AttackAssessment
    decision: DefenseDecision
    progression: AttackProgressionReport
    authorized_cut_points: list[DefensiveCutPoint]
    withheld_cut_points: list[DefensiveCutPoint]
    critical_entity_ids: list[str]
    rationale: list[str]


def _corroborating_evidence(graph: CyberStateGraph, active_relation_ids: set[str]) -> int:
    evidence_ids: set[str] = set()
    for relation in graph.relations:
        if relation.relation_id not in active_relation_ids:
            continue
        if relation.status not in {EvidenceStatus.OBSERVED, EvidenceStatus.INFERRED}:
            continue
        evidence_ids.update(item.evidence_id for item in relation.evidence)
    return len(evidence_ids)


def _critical_asset_at_risk(
    graph: CyberStateGraph,
    progression: AttackProgressionReport,
    critical_entity_ids: set[str],
) -> bool:
    if not critical_entity_ids:
        return False
    active_ids = set(progression.active_relation_ids)
    for relation in graph.relations:
        if relation.relation_id not in active_ids:
            continue
        if (
            relation.source_entity_id in critical_entity_ids
            or relation.target_entity_id in critical_entity_ids
        ):
            return True
    return any(
        cut.entity_id in critical_entity_ids for cut in progression.defensive_cut_points
    )


def _evidence_gap(graph: CyberStateGraph, progression: AttackProgressionReport) -> bool:
    active = set(progression.active_relation_ids)
    relevant = [relation for relation in graph.relations if relation.relation_id in active]
    if not relevant:
        return True
    observed = sum(relation.status is EvidenceStatus.OBSERVED for relation in relevant)
    inferred = sum(relation.status is EvidenceStatus.INFERRED for relation in relevant)
    return observed == 0 or inferred > observed


def build_active_defense_plan(
    graph: CyberStateGraph,
    *,
    critical_entity_ids: list[str] | None = None,
    focus_entity_ids: list[str] | None = None,
) -> ActiveDefensePlan:
    critical = set(critical_entity_ids or [])
    focus = set(focus_entity_ids or [])
    known_entities = {entity.entity_id for entity in graph.entities}
    unknown_critical = sorted(critical - known_entities)
    if unknown_critical:
        raise ValueError(
            "critical_entity_ids reference unknown graph entities: "
            + ", ".join(unknown_critical)
        )
    unknown_focus = sorted(focus - known_entities)
    if unknown_focus:
        raise ValueError(
            "focus_entity_ids reference unknown graph entities: "
            + ", ".join(unknown_focus)
        )

    progression = analyze_attack_progression(
        graph,
        focus_entity_ids=sorted(focus or critical),
    )
    active_ids = set(progression.active_relation_ids)
    evidence_count = _corroborating_evidence(graph, active_ids)
    critical_at_risk = _critical_asset_at_risk(graph, progression, critical)
    active_stages = [row for row in progression.active_stages if row.score >= 0.5]
    active_progression = len(active_stages) >= 2 or (
        bool(active_stages) and bool(progression.predicted_transitions)
    )
    blast_radius = max(
        (row.effect_score for row in progression.defensive_cut_points),
        default=0.0,
    )

    assessment = AttackAssessment(
        assessment_id=f"assessment:{graph.graph_id}",
        attack_confidence=progression.progression_confidence,
        corroborating_evidence_count=evidence_count,
        critical_asset_at_risk=critical_at_risk,
        active_progression=active_progression,
        blast_radius_score=blast_radius,
        evidence_gap=_evidence_gap(graph, progression),
    )
    decision = decide_defense_mode(assessment)
    permitted = set(decision.permitted_actions)
    authorized = [
        row for row in progression.defensive_cut_points if row.action in permitted
    ]
    withheld = [
        row for row in progression.defensive_cut_points if row.action not in permitted
    ]

    rationale = list(decision.rationale)
    if progression.primary_component_id is not None:
        rationale.append(
            "attack progression and cut points are scoped to one primary connected attack component"
        )
    if focus:
        rationale.append("component selection was explicitly constrained by focus entities")
    if len(progression.components) > 1:
        rationale.append(
            f"{len(progression.components)} independent active components were separated before defense planning"
        )
    if critical_at_risk:
        rationale.append("active graph path touches a declared critical protected asset")
    if authorized:
        rationale.append(
            "defensive cut points are restricted to actions permitted by the selected mode"
        )
    if withheld:
        rationale.append(
            "higher-impact cut points remain withheld because the selected mode does not permit them"
        )

    return ActiveDefensePlan(
        graph_id=graph.graph_id,
        assessment=assessment,
        decision=decision,
        progression=progression,
        authorized_cut_points=authorized,
        withheld_cut_points=withheld,
        critical_entity_ids=sorted(critical),
        rationale=rationale,
    )
