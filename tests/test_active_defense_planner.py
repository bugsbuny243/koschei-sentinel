from koschei_sentinel.active_defense_planner import build_active_defense_plan
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import DefenseActionType, DefenseMode

SHA = "b" * 64


def ev(value: str) -> EvidenceRef:
    return EvidenceRef(evidence_id=value, source="planner-test", content_sha256=SHA)


def test_high_confidence_critical_attack_enters_combat_or_siege() -> None:
    graph = CyberStateGraph(
        graph_id="active-defense-001",
        entities=[
            CyberEntity(entity_id="credential:prod", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:runner", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="pipeline:prod", entity_type=CyberEntityType.PIPELINE),
            CyberEntity(entity_id="wallet:treasury", entity_type=CyberEntityType.WALLET),
            CyberEntity(entity_id="transaction:pending", entity_type=CyberEntityType.TRANSACTION),
        ],
        relations=[
            CyberRelation(
                relation_id="c1",
                source_entity_id="credential:prod",
                target_entity_id="device:runner",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("e1"), ev("e2")],
            ),
            CyberRelation(
                relation_id="c2",
                source_entity_id="device:runner",
                target_entity_id="pipeline:prod",
                relation_type="modifies_pipeline",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("e3")],
            ),
            CyberRelation(
                relation_id="c3",
                source_entity_id="pipeline:prod",
                target_entity_id="wallet:treasury",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("e4")],
            ),
            CyberRelation(
                relation_id="c4",
                source_entity_id="wallet:treasury",
                target_entity_id="transaction:pending",
                relation_type="prepares_transaction",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[ev("e5")],
            ),
        ],
    )

    plan = build_active_defense_plan(
        graph,
        critical_entity_ids=["wallet:treasury", "pipeline:prod"],
    )

    assert plan.decision.mode in {DefenseMode.COMBAT, DefenseMode.SIEGE}
    assert plan.assessment.critical_asset_at_risk is True
    assert plan.assessment.corroborating_evidence_count == 5
    actions = {row.action for row in plan.authorized_cut_points}
    assert DefenseActionType.REVOKE_CREDENTIAL in actions
    assert DefenseActionType.PAUSE_PIPELINE in actions
    assert DefenseActionType.FREEZE_SIGNER in actions


def test_uncorroborated_signal_stays_guard_and_withholds_destructive_containment() -> None:
    graph = CyberStateGraph(
        graph_id="active-defense-guard",
        entities=[
            CyberEntity(entity_id="process:suspect", entity_type=CyberEntityType.PROCESS),
            CyberEntity(entity_id="device:laptop", entity_type=CyberEntityType.DEVICE),
        ],
        relations=[
            CyberRelation(
                relation_id="g1",
                source_entity_id="process:suspect",
                target_entity_id="device:laptop",
                relation_type="executes",
                status=EvidenceStatus.INFERRED,
                confidence=0.71,
                evidence=[ev("single")],
            )
        ],
    )

    plan = build_active_defense_plan(graph, critical_entity_ids=["device:laptop"])

    assert plan.decision.mode is DefenseMode.GUARD
    assert plan.authorized_cut_points == []
    assert any(row.action is DefenseActionType.KILL_PROCESS for row in plan.withheld_cut_points)
