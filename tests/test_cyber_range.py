from koschei_sentinel.cyber_range import (
    CyberRangeScenario,
    ScenarioTruth,
    run_cyber_range_scenario,
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


def _evidence(name: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=name,
        source="range-fixture",
        content_sha256=("a" if name.endswith("1") else "b") * 64,
    )


def _attack_graph_one() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="range:reroute",
        entities=[
            CyberEntity(entity_id="cred:one", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:one", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="wallet:treasury", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:cred-device",
                source_entity_id="cred:one",
                target_entity_id="device:one",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.97,
                evidence=[_evidence("ev1")],
            ),
            CyberRelation(
                relation_id="rel:device-wallet",
                source_entity_id="device:one",
                target_entity_id="wallet:treasury",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.96,
                evidence=[_evidence("ev2")],
            ),
        ],
    )


def _attack_graph_reroute() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="range:reroute",
        entities=[
            CyberEntity(entity_id="cred:two", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="cloud:runner", entity_type=CyberEntityType.CLOUD_RESOURCE),
            CyberEntity(entity_id="wallet:treasury", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:cred-cloud",
                source_entity_id="cred:two",
                target_entity_id="cloud:runner",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.98,
                evidence=[_evidence("ev1")],
            ),
            CyberRelation(
                relation_id="rel:cloud-wallet",
                source_entity_id="cloud:runner",
                target_entity_id="wallet:treasury",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.97,
                evidence=[_evidence("ev2")],
            ),
        ],
    )


def _benign_graph() -> CyberStateGraph:
    return CyberStateGraph(
        graph_id="range:benign",
        entities=[
            CyberEntity(entity_id="identity:alice", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="device:laptop", entity_type=CyberEntityType.DEVICE),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:normal-auth",
                source_entity_id="identity:alice",
                target_entity_id="device:laptop",
                relation_type="authenticates_to",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[_evidence("ev1")],
            )
        ],
    )


def test_range_contains_and_detects_attacker_reroute() -> None:
    scenario = CyberRangeScenario(
        scenario_id="reroute-001",
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[_attack_graph_one(), _attack_graph_reroute()],
        critical_entity_ids=["wallet:treasury"],
        expected_reroute_ticks=[1],
    )

    report = run_cyber_range_scenario(scenario)

    assert report.passed is True
    assert report.containment_tick == 0
    assert report.reroutes_detected == 1
    assert report.reroute_detection_rate == 1.0
    assert report.ticks[0].defense_mode in {DefenseMode.COMBAT, DefenseMode.SIEGE}
    assert report.ticks[1].reroute_detected is True


def test_range_does_not_high_impact_contain_single_evidence_benign_event() -> None:
    scenario = CyberRangeScenario(
        scenario_id="benign-001",
        truth=ScenarioTruth.BENIGN,
        graph_snapshots=[_benign_graph()],
        critical_entity_ids=["device:laptop"],
    )

    report = run_cyber_range_scenario(scenario)

    assert report.passed is True
    assert report.false_positive_high_impact_actions == 0
    assert report.ticks[0].defense_mode is DefenseMode.GUARD
