from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import Field

from koschei_sentinel.active_defense_planner import ActiveDefensePlan, build_active_defense_plan
from koschei_sentinel.cyber_state_graph import CyberStateGraph, EvidenceStatus
from koschei_sentinel.defense_authority import (
    DefenseActionType,
    DefenseDecision,
    DefenseMode,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.perception_assurance import PerceptionAssuranceSummary
from koschei_sentinel.perception_graph_binding import (
    PerceptionGraphReceipt,
    cyber_state_graph_sha256,
    verify_perception_graph_receipt,
)
from koschei_sentinel.perception_source_registry import (
    PerceptionSourceRegistry,
    perception_registry_sha256,
)


class ActiveDefenseAssurancePolicy(StrictModel):
    schema_version: Literal["sentinel.active-defense-assurance-policy.v1"] = (
        "sentinel.active-defense-assurance-policy.v1"
    )
    min_independent_domains_combat: int = Field(default=2, ge=2, le=16)
    min_independent_domains_siege: int = Field(default=3, ge=3, le=16)


class ActiveDefenseAssuranceReceipt(StrictModel):
    schema_version: Literal["sentinel.active-defense-assurance-receipt.v1"] = (
        "sentinel.active-defense-assurance-receipt.v1"
    )
    graph_id: str
    graph_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_batch_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    base_mode: DefenseMode
    effective_mode: DefenseMode
    assurance_relation_ids: list[str]
    active_source_principals: list[str]
    active_independence_domains: list[str]
    active_independent_domain_count: int = Field(ge=0)
    unknown_active_evidence_sources: list[str]
    downgraded: bool
    high_impact_authorized: bool


class AssuredActiveDefensePlan(StrictModel):
    schema_version: Literal["sentinel.assured-active-defense-plan.v1"] = (
        "sentinel.assured-active-defense-plan.v1"
    )
    active_defense_plan: ActiveDefensePlan
    assurance: ActiveDefenseAssuranceReceipt
    policy: ActiveDefenseAssurancePolicy


_ACTIONS_BY_MODE = {
    DefenseMode.GUARD: [
        DefenseActionType.OBSERVE,
        DefenseActionType.COLLECT_EVIDENCE,
        DefenseActionType.BLOCK_IOC,
    ],
    DefenseMode.COMBAT: [
        DefenseActionType.OBSERVE,
        DefenseActionType.COLLECT_EVIDENCE,
        DefenseActionType.BLOCK_IOC,
        DefenseActionType.REVOKE_CREDENTIAL,
        DefenseActionType.TERMINATE_SESSION,
        DefenseActionType.KILL_PROCESS,
        DefenseActionType.ISOLATE_ENDPOINT,
        DefenseActionType.QUARANTINE_WORKLOAD,
        DefenseActionType.PAUSE_PIPELINE,
        DefenseActionType.FREEZE_SIGNER,
        DefenseActionType.HOLD_TRANSACTION,
    ],
    DefenseMode.SIEGE: [
        DefenseActionType.OBSERVE,
        DefenseActionType.COLLECT_EVIDENCE,
        DefenseActionType.BLOCK_IOC,
        DefenseActionType.REVOKE_CREDENTIAL,
        DefenseActionType.TERMINATE_SESSION,
        DefenseActionType.KILL_PROCESS,
        DefenseActionType.ISOLATE_ENDPOINT,
        DefenseActionType.QUARANTINE_WORKLOAD,
        DefenseActionType.PAUSE_PIPELINE,
        DefenseActionType.FREEZE_SIGNER,
        DefenseActionType.HOLD_TRANSACTION,
        DefenseActionType.ENABLE_EMERGENCY_POLICY,
    ],
}
_MODE_RANK = {
    DefenseMode.GUARD: 0,
    DefenseMode.COMBAT: 1,
    DefenseMode.SIEGE: 2,
}


def _validate_assurance_binding(
    *,
    graph: CyberStateGraph,
    graph_receipt: PerceptionGraphReceipt,
    perception_assurance: PerceptionAssuranceSummary,
    registry: PerceptionSourceRegistry,
) -> None:
    verify_perception_graph_receipt(graph, graph_receipt)
    if graph_receipt.source_batch_sha256 != perception_assurance.fused_batch_sha256:
        raise ValueError("perception assurance belongs to a different fused batch")
    registry_sha = perception_registry_sha256(registry)
    if perception_assurance.registry_id != registry.registry_id:
        raise ValueError("perception assurance registry_id does not match registry")
    if perception_assurance.registry_sha256 != registry_sha:
        raise ValueError("perception assurance registry digest does not match registry")

    enrollment_by_principal = {row.principal: row for row in registry.enrollments}
    expected_domains: set[str] = set()
    for principal in perception_assurance.source_principals:
        enrollment = enrollment_by_principal.get(principal)
        if enrollment is None or not enrollment.enabled:
            raise ValueError(f"perception assurance references unavailable source: {principal}")
        expected_domains.add(enrollment.independence_domain)
    if sorted(expected_domains) != perception_assurance.independence_domains:
        raise ValueError("perception assurance independence domains do not match registry")
    if len(expected_domains) != perception_assurance.independent_domain_count:
        raise ValueError("perception assurance independent domain count is inconsistent")


def _assurance_component_relation_ids(
    graph: CyberStateGraph,
    plan: ActiveDefensePlan,
) -> set[str]:
    active_ids = set(plan.progression.active_relation_ids)
    active_relations = [row for row in graph.relations if row.relation_id in active_ids]
    if not active_relations:
        return set()

    adjacency: dict[str, set[str]] = {}
    touched: set[str] = set()
    for relation in active_relations:
        touched.update({relation.source_entity_id, relation.target_entity_id})
        adjacency.setdefault(relation.source_entity_id, set()).add(relation.target_entity_id)
        adjacency.setdefault(relation.target_entity_id, set()).add(relation.source_entity_id)

    roots = [entity_id for entity_id in plan.critical_entity_ids if entity_id in touched]
    if not roots and plan.progression.defensive_cut_points:
        top = plan.progression.defensive_cut_points[0].entity_id
        if top in touched:
            roots = [top]
    if not roots:
        return active_ids

    component = set(roots)
    queue: deque[str] = deque(roots)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in component:
                component.add(neighbor)
                queue.append(neighbor)

    return {
        relation.relation_id
        for relation in active_relations
        if relation.source_entity_id in component and relation.target_entity_id in component
    }


def _active_evidence_domains(
    *,
    graph: CyberStateGraph,
    plan: ActiveDefensePlan,
    registry: PerceptionSourceRegistry,
) -> tuple[list[str], list[str], list[str], list[str]]:
    assurance_relation_ids = _assurance_component_relation_ids(graph, plan)
    enrollment_by_principal = {row.principal: row for row in registry.enrollments if row.enabled}
    principals: set[str] = set()
    domains: set[str] = set()
    unknown: set[str] = set()

    for relation in graph.relations:
        if relation.relation_id not in assurance_relation_ids:
            continue
        if relation.status not in {EvidenceStatus.OBSERVED, EvidenceStatus.INFERRED}:
            continue
        for evidence in relation.evidence:
            enrollment = enrollment_by_principal.get(evidence.source)
            if enrollment is None:
                unknown.add(evidence.source)
                continue
            principals.add(evidence.source)
            domains.add(enrollment.independence_domain)
    return (
        sorted(assurance_relation_ids),
        sorted(principals),
        sorted(domains),
        sorted(unknown),
    )


def _maximum_mode_for_domains(
    count: int,
    policy: ActiveDefenseAssurancePolicy,
) -> DefenseMode:
    if count >= policy.min_independent_domains_siege:
        return DefenseMode.SIEGE
    if count >= policy.min_independent_domains_combat:
        return DefenseMode.COMBAT
    return DefenseMode.GUARD


def build_assured_active_defense_plan(
    graph: CyberStateGraph,
    *,
    graph_receipt: PerceptionGraphReceipt,
    perception_assurance: PerceptionAssuranceSummary,
    registry: PerceptionSourceRegistry,
    critical_entity_ids: list[str] | None = None,
    policy: ActiveDefenseAssurancePolicy | None = None,
) -> AssuredActiveDefensePlan:
    gate = policy or ActiveDefenseAssurancePolicy()
    _validate_assurance_binding(
        graph=graph,
        graph_receipt=graph_receipt,
        perception_assurance=perception_assurance,
        registry=registry,
    )
    base = build_active_defense_plan(
        graph,
        critical_entity_ids=critical_entity_ids,
    )
    relation_ids, active_principals, active_domains, unknown_sources = _active_evidence_domains(
        graph=graph,
        plan=base,
        registry=registry,
    )
    maximum_mode = _maximum_mode_for_domains(len(active_domains), gate)
    effective_mode = min(
        (base.decision.mode, maximum_mode),
        key=lambda mode: _MODE_RANK[mode],
    )
    downgraded = effective_mode is not base.decision.mode

    permitted_actions = _ACTIONS_BY_MODE[effective_mode]
    permitted_set = set(permitted_actions)
    authorized = [
        row for row in base.progression.defensive_cut_points if row.action in permitted_set
    ]
    withheld = [
        row for row in base.progression.defensive_cut_points if row.action not in permitted_set
    ]

    decision_rationale = list(base.decision.rationale)
    decision_rationale.append(
        f"relevant attack component spans {len(active_domains)} admitted independence domain(s)"
    )
    if downgraded:
        decision_rationale.append(
            f"assurance gate downgraded {base.decision.mode.value} to {effective_mode.value}"
        )
    if unknown_sources:
        decision_rationale.append(
            "unregistered or non-admitted evidence sources were excluded from active corroboration"
        )
    decision = DefenseDecision(
        mode=effective_mode,
        permitted_actions=permitted_actions,
        rationale=decision_rationale,
    )

    plan_rationale = list(base.rationale)
    if downgraded:
        plan_rationale.append(
            "higher-impact containment is withheld until independent evidence in the same attack component is sufficient"
        )
    adjusted = ActiveDefensePlan(
        graph_id=base.graph_id,
        assessment=base.assessment,
        decision=decision,
        progression=base.progression,
        authorized_cut_points=authorized,
        withheld_cut_points=withheld,
        critical_entity_ids=base.critical_entity_ids,
        rationale=plan_rationale,
    )
    assurance_receipt = ActiveDefenseAssuranceReceipt(
        graph_id=graph.graph_id,
        graph_sha256=cyber_state_graph_sha256(graph),
        source_batch_sha256=graph_receipt.source_batch_sha256,
        registry_sha256=perception_assurance.registry_sha256,
        base_mode=base.decision.mode,
        effective_mode=effective_mode,
        assurance_relation_ids=relation_ids,
        active_source_principals=active_principals,
        active_independence_domains=active_domains,
        active_independent_domain_count=len(active_domains),
        unknown_active_evidence_sources=unknown_sources,
        downgraded=downgraded,
        high_impact_authorized=effective_mode is not DefenseMode.GUARD,
    )
    return AssuredActiveDefensePlan(
        active_defense_plan=adjusted,
        assurance=assurance_receipt,
        policy=gate,
    )
