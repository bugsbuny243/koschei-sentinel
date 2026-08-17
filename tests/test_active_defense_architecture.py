import hashlib

import pytest

from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import (
    AttackAssessment,
    DefenseExecutionRecord,
    DefenseMode,
    decide_defense_mode,
)


def _evidence(evidence_id: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source="unit-test",
        content_sha256=hashlib.sha256(evidence_id.encode()).hexdigest(),
    )


def test_observed_relation_requires_evidence():
    with pytest.raises(ValueError):
        CyberRelation(
            relation_id="r-1",
            source_entity_id="identity:alice",
            target_entity_id="device:1",
            relation_type="authenticated_to",
            status=EvidenceStatus.OBSERVED,
            confidence=1.0,
        )


def test_state_graph_links_identity_to_pipeline():
    graph = CyberStateGraph(
        graph_id="incident-1",
        entities=[
            CyberEntity(entity_id="identity:alice", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="device:1", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="pipeline:prod", entity_type=CyberEntityType.PIPELINE),
        ],
        relations=[
            CyberRelation(
                relation_id="r-1",
                source_entity_id="identity:alice",
                target_entity_id="device:1",
                relation_type="authenticated_to",
                status=EvidenceStatus.OBSERVED,
                confidence=1.0,
                evidence=[_evidence("ev-1")],
            ),
            CyberRelation(
                relation_id="r-2",
                source_entity_id="device:1",
                target_entity_id="pipeline:prod",
                relation_type="accessed",
                status=EvidenceStatus.INFERRED,
                confidence=0.88,
                evidence=[_evidence("ev-2")],
            ),
        ],
    )
    assert len(graph.entities) == 3
    assert len(graph.relations) == 2


def test_combat_mode_requires_corroboration():
    decision = decide_defense_mode(
        AttackAssessment(
            assessment_id="attack-1",
            attack_confidence=0.92,
            corroborating_evidence_count=2,
            active_progression=True,
        )
    )
    assert decision.mode is DefenseMode.COMBAT


def test_siege_mode_requires_critical_active_broad_attack():
    decision = decide_defense_mode(
        AttackAssessment(
            assessment_id="attack-2",
            attack_confidence=0.98,
            corroborating_evidence_count=4,
            critical_asset_at_risk=True,
            active_progression=True,
            blast_radius_score=0.9,
        )
    )
    assert decision.mode is DefenseMode.SIEGE


def test_authorized_execution_requires_evidence():
    with pytest.raises(ValueError):
        DefenseExecutionRecord(
            action="ISOLATE_ENDPOINT",
            target="device:1",
            mode=DefenseMode.COMBAT,
            authorized=True,
        )
