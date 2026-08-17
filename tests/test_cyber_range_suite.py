from koschei_sentinel.cyber_range import CyberRangeScenario, ScenarioTruth
from koschei_sentinel.cyber_range_suite import CyberRangeGatePolicy, run_cyber_range_suite
from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)


def _ev(name: str, char: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=name,
        source="range-suite-fixture",
        content_sha256=char * 64,
    )


def _attack_graph(graph_id: str) -> CyberStateGraph:
    return CyberStateGraph(
        graph_id=graph_id,
        entities=[
            CyberEntity(entity_id="cred:test", entity_type=CyberEntityType.CREDENTIAL),
            CyberEntity(entity_id="device:test", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="wallet:test", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:one",
                source_entity_id="cred:test",
                target_entity_id="device:test",
                relation_type="uses_credential",
                status=EvidenceStatus.OBSERVED,
                confidence=0.98,
                evidence=[_ev("evidence:one", "a")],
            ),
            CyberRelation(
                relation_id="rel:two",
                source_entity_id="device:test",
                target_entity_id="wallet:test",
                relation_type="reaches_signer",
                status=EvidenceStatus.OBSERVED,
                confidence=0.98,
                evidence=[_ev("evidence:two", "b")],
            ),
        ],
    )


def _malicious() -> CyberRangeScenario:
    return CyberRangeScenario(
        scenario_id="suite-malicious",
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[_attack_graph("suite:malicious")],
        critical_entity_ids=["wallet:test"],
    )


def _delayed_malicious() -> CyberRangeScenario:
    graph_id = "suite:delayed"
    initial = CyberStateGraph(
        graph_id=graph_id,
        entities=[
            CyberEntity(entity_id="identity:test", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="device:test", entity_type=CyberEntityType.DEVICE),
            CyberEntity(entity_id="wallet:test", entity_type=CyberEntityType.WALLET),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:initial",
                source_entity_id="identity:test",
                target_entity_id="device:test",
                relation_type="authenticates_to",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[_ev("evidence:initial", "d")],
            )
        ],
    )
    return CyberRangeScenario(
        scenario_id="suite-delayed-malicious",
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[initial, _attack_graph(graph_id)],
        critical_entity_ids=["wallet:test"],
    )


def _benign() -> CyberRangeScenario:
    graph = CyberStateGraph(
        graph_id="suite:benign",
        entities=[
            CyberEntity(entity_id="identity:test", entity_type=CyberEntityType.IDENTITY),
            CyberEntity(entity_id="device:normal", entity_type=CyberEntityType.DEVICE),
        ],
        relations=[
            CyberRelation(
                relation_id="rel:normal",
                source_entity_id="identity:test",
                target_entity_id="device:normal",
                relation_type="authenticates_to",
                status=EvidenceStatus.OBSERVED,
                confidence=0.99,
                evidence=[_ev("evidence:normal", "c")],
            )
        ],
    )
    return CyberRangeScenario(
        scenario_id="suite-benign",
        truth=ScenarioTruth.BENIGN,
        graph_snapshots=[graph],
        critical_entity_ids=["device:normal"],
    )


def test_suite_passes_containment_and_benign_safety_gates() -> None:
    report = run_cyber_range_suite([_malicious(), _benign()])

    assert report.passed is True
    assert report.malicious_containment_rate == 1.0
    assert report.benign_high_impact_false_positive_rate == 0.0


def test_suite_fails_a_stricter_latency_gate() -> None:
    policy = CyberRangeGatePolicy(mean_containment_tick_max=0.0)
    report = run_cyber_range_suite([_delayed_malicious()], policy=policy)

    assert report.passed is False
    assert report.mean_containment_tick == 1.0
    assert any("mean containment tick above gate" in item for item in report.violations)
