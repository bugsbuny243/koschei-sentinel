from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.multi_incident_defense import build_multi_incident_defense_plan

SHA = "e" * 64


def ev(name: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"evidence:{name}",
        source="ENDPOINT:edr:test",
        content_sha256=SHA,
    )


def _graph() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="incident:parallel",
        entities=[
            CyberEntity(entity_id="identity:user", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="device:laptop", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="process:malware", entity_type=CyberEntityType.PROCESS),
            CyberEntity(entity_id="credential:prod", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="pipeline:prod", entity_type=CyberEntityType.PIPELINE),
            CyberEntity(entity_id="wallet:treasury", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="left-1",
                source_entity_id="identity:user",
                target_entity_id="device:laptop",
                relation_type="authenticates_to",
                status=EvidenceStatus.OBSERVED,
                confidence=0.95,
                evidence=[ev("left-1")],
            ),
            CyberRelation(
                relation_id="left-2",
                source_entity_id="device:laptop",
                target_entity_id="process:malware",
                relation_type="executes",
                status=EvidenceStatus.OBSERVED,
                confidence=0.95,
                evidence=[ev("left-2")],
            ),
            CyberRelation(
                relation_id="right-1",
                source_entity_id="credential:prod",
                target_entity_id="pipeline:prod",
                relation_type="modifies_pipeline",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("right-1")],
            ),
            CyberRelation(
                relation_id="right-2",
                source_entity_id="pipeline:prod",
                target_entity_id="wallet:treasury",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("right-2")],
            ),
        ],
    )


def test_parallel_attack_components_receive_independent_defense_plans() -> None:
    plan = build_multi_incident_defense_plan(
        _graph(),
        critical_entity_ids=["device:laptop", "wallet:treasury"],
    )

    assert plan.active_component_count == 2
    assert plan.critical_component_count == 2
    assert len(plan.component_plans) == 2
    assert len({row.component_id for row in plan.component_plans}) == 2

    for component_plan in plan.component_plans:
        component_entities = set(component_plan.entity_ids)
        cut_entities = {
            row.entity_id
            for row in (
                component_plan.defense_plan.authorized_cut_points
                + component_plan.defense_plan.withheld_cut_points
            )
        }
        assert cut_entities <= component_entities


def test_inactive_critical_assets_are_not_invented_as_attack_components() -> None:
    graph = _graph()
    graph.entities.append(
        CyberEntity(entity_id="wallet:cold", entity_type=CyberEntityType.WALLET)
    )

    plan = build_multi_incident_defense_plan(
        graph,
        critical_entity_ids=["wallet:cold"],
    )

    assert plan.active_component_count == 2
    assert plan.critical_component_count == 0
    assert plan.inactive_critical_entity_ids == ["wallet:cold"]


def test_multi_incident_priority_order_is_deterministic() -> None:
    first = build_multi_incident_defense_plan(
        _graph(),
        critical_entity_ids=["device:laptop", "wallet:treasury"],
    )
    second = build_multi_incident_defense_plan(
        _graph(),
        critical_entity_ids=["device:laptop", "wallet:treasury"],
    )

    assert [row.component_id for row in first.component_plans] == [
        row.component_id for row in second.component_plans
    ]
    assert [row.priority_score for row in first.component_plans] == [
        row.priority_score for row in second.component_plans
    ]
