from koschei_sentinel.attack_world_lines import WorldLineTransitionType
from koschei_sentinel.cyber_range import SimulatedActionOutcome
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.defense_authority import DefenseActionType
from koschei_sentinel.multi_incident_cyber_range import (
    ExpectedWorldLineTransition,
    MultiIncidentCyberRangeScenario,
    MultiIncidentScenarioTruth,
    run_multi_incident_cyber_range,
)

SHA = "f" * 64


def ev(name: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"evidence:{name}",
        source="range:test",
        content_sha256=SHA,
    )


def _snapshot_one() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="multi-range:incident",
        entities=[
            CyberEntity(entity_id="credential:a", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:a", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="pipeline:a", entity_type=CyberEntityType.PIPELINE),
            CyberEntity(entity_id="wallet:a", entity_type=CyberEntityType.WALLET),
            CyberEntity(entity_id="credential:b", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:b", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="process:b", entity_type=CyberEntityType.PROCESS),
        ],
        relations=[
            CyberRelation(relation_id="rel:a1", source_entity_id="credential:a", target_entity_id="device:a", relation_type="uses_credential", status=EvidenceStatus.OBSERVED, confidence=0.99, evidence=[ev("a1")]),
            CyberRelation(relation_id="rel:a2", source_entity_id="device:a", target_entity_id="pipeline:a", relation_type="modifies_pipeline", status=EvidenceStatus.OBSERVED, confidence=0.99, evidence=[ev("a2")]),
            CyberRelation(relation_id="rel:a3", source_entity_id="pipeline:a", target_entity_id="wallet:a", relation_type="reaches_signer", status=EvidenceStatus.OBSERVED, confidence=0.99, evidence=[ev("a3")]),
            CyberRelation(relation_id="rel:b1", source_entity_id="credential:b", target_entity_id="device:b", relation_type="uses_credential", status=EvidenceStatus.OBSERVED, confidence=0.95, evidence=[ev("b1")]),
            CyberRelation(relation_id="rel:b2", source_entity_id="device:b", target_entity_id="process:b", relation_type="executes", status=EvidenceStatus.OBSERVED, confidence=0.95, evidence=[ev("b2")]),
        ],
    )


def _snapshot_two() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="multi-range:incident",
        entities=[
            CyberEntity(entity_id="credential:a2", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:a2", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="pipeline:a2", entity_type=CyberEntityType.PIPELINE),
            CyberEntity(entity_id="wallet:a", entity_type=CyberEntityType.WALLET),
            CyberEntity(entity_id="credential:b", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:b", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="process:b", entity_type=CyberEntityType.PROCESS),
        ],
        relations=[
            CyberRelation(relation_id="rel:a4", source_entity_id="credential:a2", target_entity_id="device:a2", relation_type="uses_credential", status=EvidenceStatus.OBSERVED, confidence=0.99, evidence=[ev("a4")]),
            CyberRelation(relation_id="rel:a5", source_entity_id="device:a2", target_entity_id="pipeline:a2", relation_type="modifies_pipeline", status=EvidenceStatus.OBSERVED, confidence=0.99, evidence=[ev("a5")]),
            CyberRelation(relation_id="rel:a6", source_entity_id="pipeline:a2", target_entity_id="wallet:a", relation_type="reaches_signer", status=EvidenceStatus.OBSERVED, confidence=0.99, evidence=[ev("a6")]),
            CyberRelation(relation_id="rel:b1", source_entity_id="credential:b", target_entity_id="device:b", relation_type="uses_credential", status=EvidenceStatus.OBSERVED, confidence=0.95, evidence=[ev("b1")]),
            CyberRelation(relation_id="rel:b2", source_entity_id="device:b", target_entity_id="process:b", relation_type="executes", status=EvidenceStatus.OBSERVED, confidence=0.95, evidence=[ev("b2")]),
        ],
    )


def test_multi_incident_range_tracks_parallel_attacks_and_reroute_without_leakage() -> None:
    scenario = MultiIncidentCyberRangeScenario(
        scenario_id="multi-range-golden",
        truth=MultiIncidentScenarioTruth.MALICIOUS,
        graph_snapshots=[_snapshot_one(), _snapshot_two()],
        critical_entity_ids=["wallet:a", "device:b"],
        expected_active_component_counts=[2, 2],
        expected_world_line_transitions=[
            ExpectedWorldLineTransition(to_tick=1, transition_type=WorldLineTransitionType.REROUTED, protected_anchor_id="wallet:a"),
            ExpectedWorldLineTransition(to_tick=1, transition_type=WorldLineTransitionType.CONTINUED, protected_anchor_id="device:b"),
        ],
        action_outcomes=[
            SimulatedActionOutcome(action=DefenseActionType.FREEZE_SIGNER, target_entity_id="wallet:a", succeeds=True),
            SimulatedActionOutcome(action=DefenseActionType.ISOLATE_ENDPOINT, target_entity_id="device:b", succeeds=True),
        ],
    )
    report = run_multi_incident_cyber_range(scenario)
    assert report.component_count_accuracy == 1.0
    assert report.cut_point_leakage_count == 0
    assert report.world_line_transition_accuracy == 1.0
    assert report.missed_world_line_transitions == []
    assert all(tick.active_component_count == 2 for tick in report.ticks)
    assert report.passed is True


def test_wrong_expected_component_count_fails_gate() -> None:
    scenario = MultiIncidentCyberRangeScenario(
        scenario_id="multi-range-count-fail",
        truth=MultiIncidentScenarioTruth.MIXED,
        graph_snapshots=[_snapshot_one()],
        critical_entity_ids=["wallet:a"],
        expected_active_component_counts=[1],
    )
    report = run_multi_incident_cyber_range(scenario)
    assert report.component_count_accuracy == 0.0
    assert report.passed is False
    assert any("component counts" in value for value in report.violations)


def test_missing_expected_world_line_transition_fails_gate() -> None:
    scenario = MultiIncidentCyberRangeScenario(
        scenario_id="multi-range-lineage-fail",
        truth=MultiIncidentScenarioTruth.MIXED,
        graph_snapshots=[_snapshot_one(), _snapshot_two()],
        critical_entity_ids=["wallet:a"],
        expected_world_line_transitions=[
            ExpectedWorldLineTransition(to_tick=1, transition_type=WorldLineTransitionType.MERGED, protected_anchor_id="wallet:a")
        ],
    )
    report = run_multi_incident_cyber_range(scenario)
    assert report.world_line_transition_accuracy == 0.0
    assert report.missed_world_line_transitions
    assert report.passed is False
