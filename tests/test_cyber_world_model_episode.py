from koschei_sentinel.attack_world_lines import WorldLineTransitionType
from koschei_sentinel.cyber_range import CyberRangeScenario, ScenarioTruth, run_cyber_range_scenario
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.cyber_world_model_episode import (
    TemporalTransitionType,
    build_world_model_episode,
)


def _evidence(name: str, char: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=name,
        source="world-model-fixture",
        content_sha256=char * 64,
    )


def _scenario() -> CyberRangeScenario:
    first = CyberStateGraph(
        graph_id="world:incident",
        entities=[
            CyberEntity(entity_id="cred:a", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:a", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="wallet:a", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:observed",
                source_entity_id="cred:a",
                target_entity_id="device:a",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.98,
                evidence=[_evidence("ev:observed", "a")],
            ),
            CyberRelation(
                relation_id="rel:predicted",
                source_entity_id="device:a",
                target_entity_id="wallet:a",
                relation_type="reaches_signer",
                status=EvidenceStatus.PREDICTED,
                confidence=0.7,
                evidence=[],
            ),
        ],
    )
    second = CyberStateGraph(
        graph_id="world:incident",
        entities=[
            CyberEntity(entity_id="cred:b", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:b", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="wallet:a", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:reroute-one",
                source_entity_id="cred:b",
                target_entity_id="device:b",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[_evidence("ev:reroute-one", "b")],
            ),
            CyberRelation(
                relation_id="rel:reroute-two",
                source_entity_id="device:b",
                target_entity_id="wallet:a",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[_evidence("ev:reroute-two", "c")],
            ),
            CyberRelation(
                relation_id="rel:disproved",
                source_entity_id="cred:b",
                target_entity_id="wallet:a",
                relation_type="uses_wallet",
                status=EvidenceStatus.DISPROVED,
                confidence=0.1,
                evidence=[_evidence("ev:disproved", "d")],
            ),
        ],
    )
    return CyberRangeScenario(
        scenario_id="world-episode",
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[first, second],
        critical_entity_ids=["wallet:a"],
        expected_reroute_ticks=[1],
    )


def test_world_model_episode_preserves_evidence_classes_and_reroute_transition() -> None:
    scenario = _scenario()
    report = run_cyber_range_scenario(scenario)
    episode = build_world_model_episode(scenario, report)

    assert episode.training_authorization is False
    assert episode.snapshots[0].observed_relations == 1
    assert episode.snapshots[0].predicted_relations == 1
    assert episode.snapshots[1].disproved_relations == 1
    assert episode.transitions[0].transition_type is TemporalTransitionType.ATTACKER_REROUTE
    assert episode.transitions[0].graph_changed is True
    assert episode.attack_world_lines is not None
    reroute = next(
        row
        for row in episode.attack_world_lines.transitions
        if row.to_tick == 1
        and row.transition_type is WorldLineTransitionType.REROUTED
    )
    assert reroute.shared_protected_anchor_ids == ["wallet:a"]
    assert reroute.predecessor_world_line_ids == reroute.successor_world_line_ids
