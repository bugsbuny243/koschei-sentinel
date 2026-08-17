from koschei_sentinel.active_defense_planner import build_active_defense_plan
from koschei_sentinel.attack_progression import AttackStage, analyze_attack_progression
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)


SHA = "c" * 64


def ev(evidence_id: str, source: str = "ENDPOINT:edr:test") -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source=source,
        content_sha256=SHA,
    )


def _two_component_graph() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="multi-incident-001",
        entities=[
            CyberEntity(entity_id="identity:alice", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="device:laptop", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="credential:prod", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="pipeline:release", entity_type=CyberEntityType.PIPELINE),
            CyberEntity(entity_id="wallet:treasury", entity_type=CyberEntityType.WALLET),
            CyberEntity(entity_id="transaction:pending", entity_type=CyberEntityType.TRANSACTION),
        ],
        relations=[
            CyberRelation(
                relation_id="a1",
                source_entity_id="identity:alice",
                target_entity_id="device:laptop",
                relation_type="authenticates_to",
                status=EvidenceStatus.OBSERVED,
                confidence=0.91,
                evidence=[ev("ev-a1")],
            ),
            CyberRelation(
                relation_id="b1",
                source_entity_id="credential:prod",
                target_entity_id="pipeline:release",
                relation_type="modifies_pipeline",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("ev-b1", "CICD:ci:prod")],
            ),
            CyberRelation(
                relation_id="b2",
                source_entity_id="pipeline:release",
                target_entity_id="wallet:treasury",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("ev-b2", "SIGNER_WALLET:signer:prod")],
            ),
            CyberRelation(
                relation_id="b3",
                source_entity_id="wallet:treasury",
                target_entity_id="transaction:pending",
                relation_type="executes_transaction",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("ev-b3", "BLOCKCHAIN:chain:prod")],
            ),
        ],
    )


def test_disconnected_events_are_reported_as_separate_attack_components() -> None:
    report = analyze_attack_progression(_two_component_graph())

    assert len(report.components) == 2
    assert report.primary_component_id is not None
    assert report.current_stage is AttackStage.IMPACT
    assert set(report.active_relation_ids) == {"b1", "b2", "b3"}

    relation_sets = {frozenset(component.active_relation_ids) for component in report.components}
    assert relation_sets == {frozenset({"a1"}), frozenset({"b1", "b2", "b3"})}


def test_focus_entity_selects_its_component_even_when_another_component_is_higher_risk() -> None:
    report = analyze_attack_progression(
        _two_component_graph(),
        focus_entity_ids=["device:laptop"],
    )

    assert report.current_stage is AttackStage.INITIAL_ACCESS
    assert report.active_relation_ids == ["a1"]
    assert all(
        cut.entity_id in {"identity:alice", "device:laptop"}
        for cut in report.defensive_cut_points
    )


def test_active_defense_planner_does_not_mix_cut_points_across_incidents() -> None:
    plan = build_active_defense_plan(
        _two_component_graph(),
        critical_entity_ids=["device:laptop"],
    )

    assert plan.progression.active_relation_ids == ["a1"]
    assert len(plan.progression.components) == 2
    cut_entities = {
        row.entity_id
        for row in plan.authorized_cut_points + plan.withheld_cut_points
    }
    assert cut_entities <= {"identity:alice", "device:laptop"}
    assert "wallet:treasury" not in cut_entities
