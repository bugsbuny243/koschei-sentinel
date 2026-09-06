from koschei_sentinel.attack_progression import (
    AttackStage,
    analyze_attack_progression,
)
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import DefenseActionType

SHA = "a" * 64


def evidence(evidence_id: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source="test-telemetry",
        content_sha256=SHA,
    )


def test_progression_sees_supply_chain_and_signer_path() -> None:
    graph = CyberStateGraph(
        graph_id="incident-neo-001",
        entities=[
            CyberEntity(entity_id="identity:attacker-session", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="credential:ci-token", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:runner", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="pipeline:release", entity_type=CyberEntityType.PIPELINE),
            CyberEntity(entity_id="wallet:signer", entity_type=CyberEntityType.WALLET),
            CyberEntity(entity_id="transaction:pending", entity_type=CyberEntityType.TRANSACTION),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:r1",
                source_entity_id="identity:attacker-session",
                target_entity_id="credential:ci-token",
                relation_type="steals_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.98,
                evidence=[evidence("ev1")],
            ),
            CyberRelation(
                relation_id="rel:r2",
                source_entity_id="credential:ci-token",
                target_entity_id="device:runner",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.96,
                evidence=[evidence("ev2")],
            ),
            CyberRelation(
                relation_id="rel:r3",
                source_entity_id="device:runner",
                target_entity_id="pipeline:release",
                relation_type="modifies_pipeline",
                status=EvidenceStatus.OBSERVED,
                confidence=0.94,
                evidence=[evidence("ev3")],
            ),
            CyberRelation(
                relation_id="rel:r4",
                source_entity_id="pipeline:release",
                target_entity_id="wallet:signer",
                relation_type="reaches_signer",
                status=EvidenceStatus.INFERRED,
                confidence=0.90,
                evidence=[evidence("ev4")],
            ),
            CyberRelation(
                relation_id="rel:r5",
                source_entity_id="wallet:signer",
                target_entity_id="transaction:pending",
                relation_type="prepares_transaction",
                status=EvidenceStatus.PREDICTED,
                confidence=0.80,
                evidence=[],
            ),
        ],
    )

    report = analyze_attack_progression(graph)

    assert report.current_stage is AttackStage.SIGNER_OR_WALLET_ACCESS
    assert report.progression_confidence >= 0.9
    assert any(row.stage is AttackStage.SUPPLY_CHAIN for row in report.active_stages)
    assert any(row.stage is AttackStage.SIGNER_OR_WALLET_ACCESS for row in report.active_stages)
    assert report.predicted_transitions[0].to_stage is AttackStage.IMPACT
    assert "rel:r5" not in report.active_relation_ids

    actions = {row.action for row in report.defensive_cut_points}
    assert DefenseActionType.REVOKE_CREDENTIAL in actions
    assert DefenseActionType.PAUSE_PIPELINE in actions
    assert DefenseActionType.FREEZE_SIGNER in actions


def test_disproved_relation_does_not_advance_attack() -> None:
    graph = CyberStateGraph(
        graph_id="incident-disproved",
        entities=[
            CyberEntity(entity_id="device:one", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="wallet:one", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="false-signer-path",
                source_entity_id="device:one",
                target_entity_id="wallet:one",
                relation_type="reaches_signer",
                status=EvidenceStatus.DISPROVED,
                confidence=0.2,
                evidence=[evidence("ev-false")],
            )
        ],
    )

    report = analyze_attack_progression(graph)

    assert report.current_stage is None
    assert report.progression_confidence == 0.0
    assert report.defensive_cut_points == []
    assert report.disproved_relation_ids == ["false-signer-path"]
