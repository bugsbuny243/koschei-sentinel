import pytest

from koschei_sentinel.cyber_perception import (
    PerceivedEntity,
    PerceptionBatch,
    PerceptionObservation,
    TelemetrySourceType,
    compile_perception_batch,
)
from koschei_sentinel.cyber_state_graph import CyberEntityType, EvidenceStatus


def _observation(
    *,
    observation_id: str,
    evidence_id: str,
    evidence_char: str,
    device_label: str = "host-a",
) -> PerceptionObservation:
    return PerceptionObservation(
        observation_id=observation_id,
        source_type=TelemetrySourceType.CLOUD_IAM,
        source_instance="iam:test",
        source_entity=PerceivedEntity(
            entity_id="cred:test",
            entity_type=CyberEntityType.CREDENTIAL,
            labels={"kind": "temporary"},
        ),
        target_entity=PerceivedEntity(
            entity_id="device:test",
            entity_type=CyberEntityType.DEVICE,
            labels={"hostname": device_label},
        ),
        relation_type="uses_credential",
        confidence=0.95,
        evidence_id=evidence_id,
        evidence_sha256=evidence_char * 64,
    )


def test_perception_compiles_only_observed_relations_and_aggregates_evidence() -> None:
    graph = compile_perception_batch(
        PerceptionBatch(
            batch_id="batch:test",
            observations=[
                _observation(
                    observation_id="obs:one",
                    evidence_id="evidence:one",
                    evidence_char="a",
                ),
                _observation(
                    observation_id="obs:two",
                    evidence_id="evidence:two",
                    evidence_char="b",
                ),
            ],
        ),
        graph_id="incident:test",
    )

    assert len(graph.relations) == 1
    assert graph.relations[0].status is EvidenceStatus.OBSERVED
    assert len(graph.relations[0].evidence) == 2
    assert graph.relations[0].confidence == 0.95


def test_conflicting_entity_labels_fail_closed() -> None:
    batch = PerceptionBatch(
        batch_id="batch:conflict",
        observations=[
            _observation(
                observation_id="obs:one",
                evidence_id="evidence:one",
                evidence_char="a",
                device_label="host-a",
            ),
            _observation(
                observation_id="obs:two",
                evidence_id="evidence:two",
                evidence_char="b",
                device_label="host-b",
            ),
        ],
    )
    with pytest.raises(ValueError, match="conflicting entity label"):
        compile_perception_batch(batch, graph_id="incident:conflict")


def test_sensitive_material_is_rejected_from_entity_labels() -> None:
    with pytest.raises(ValueError, match="sensitive material"):
        PerceivedEntity(
            entity_id="identity:test",
            entity_type=CyberEntityType.IDENTITY,
            labels={"private_key": "must-not-enter-world-model"},
        )
