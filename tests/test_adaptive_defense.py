from koschei_sentinel.active_defense_planner import build_active_defense_plan
from koschei_sentinel.adaptive_defense import (
    ReassessmentDisposition,
    reassess_after_interception,
)
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import DefenseMode
from koschei_sentinel.interception_execution import start_interception_execution
from koschei_sentinel.interception_planner import build_interception_plan


def _evidence(evidence_id: str, digest_char: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source="edr",
        content_sha256=digest_char * 64,
    )


def test_reassessment_recomputes_authority_from_updated_graph():
    entities = [
        CyberEntity(entity_id="identity:user", entity_type=CyberEntityType.IDENTITY),
        CyberEntity(entity_id="credential:token", entity_type=CyberEntityType.CREDENTIAL),
        CyberEntity(entity_id="device:prod", entity_type=CyberEntityType.DEVICE),
    ]
    previous_graph = CyberStateGraph(graph_id="incident:1", entities=entities, relations=[])
    previous_active = build_active_defense_plan(
        previous_graph,
        critical_entity_ids=["device:prod"],
    )
    previous_interception = build_interception_plan(previous_active)
    previous_execution = start_interception_execution(previous_interception)

    updated_graph = CyberStateGraph(
        graph_id="incident:1",
        entities=entities,
        relations=[
            CyberRelation(
                relation_id="rel:initial",
                source_entity_id="identity:user",
                target_entity_id="device:prod",
                relation_type="authenticates_to",
                status=EvidenceStatus.OBSERVED,
                confidence=0.96,
                evidence=[_evidence("evidence:initial", "a")],
            ),
            CyberRelation(
                relation_id="rel:credential",
                source_entity_id="credential:token",
                target_entity_id="device:prod",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.97,
                evidence=[_evidence("evidence:credential", "b")],
            ),
        ],
    )

    reassessment = reassess_after_interception(
        previous_graph=previous_graph,
        updated_graph=updated_graph,
        previous_execution=previous_execution,
        critical_entity_ids=["device:prod"],
    )

    assert reassessment.graph_changed is True
    assert reassessment.disposition is ReassessmentDisposition.CONTINUE_CONTAINMENT
    assert reassessment.active_defense_plan.decision.mode is DefenseMode.COMBAT
    assert reassessment.interception_plan.steps
    assert any(
        step.target_entity_id == "device:prod"
        for step in reassessment.interception_plan.steps
    )
