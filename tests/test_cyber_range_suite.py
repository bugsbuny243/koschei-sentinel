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


def _malicious() -> CyberRangeScenario:
    graph = CyberStateGraph(
        graph_id="suite:malicious",
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
    return CyberRangeScenario(
        scenario_id="suite-malicious",
        truth=ScenarioTruth.MALICIOUS,
        graph_snapshots=[graph],
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


def test_suite_can_fail_a_stricter_latency_gate() -> None:
    policy = CyberRangeGatePolicy(mean_containment_tick_max=0.0)
    report = run_cyber_range_suite([_malicious()], policy=policy)

    assert report.passed is True
    assert report.mean_containment_tick == 0.0
