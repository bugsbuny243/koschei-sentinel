import pytest

from koschei_sentinel.cyber_perception import (
    PerceivedEntity,
    PerceptionBatch,
    PerceptionObservation,
    TelemetrySourceType,
    compile_perception_batch,
)
from koschei_sentinel.cyber_state_graph import CyberEntityType, EvidenceStatus
from koschei_sentinel.perception_adapter_profiles import (
    generic_cloud_iam_adapter,
    generic_endpoint_process_adapter,
)
from koschei_sentinel.perception_adapter_sdk import (
    SanitizedTelemetryEvent,
    run_perception_adapter_batch,
)
from koschei_sentinel.perception_fusion import fuse_perception_batches


def _endpoint_batch() -> PerceptionBatch:
    event = SanitizedTelemetryEvent(
        event_id="event:endpoint",
        source_type=TelemetrySourceType.ENDPOINT,
        source_instance="edr:one",
        payload_sha256="a" * 64,
        fields={
            "device_id": "device:alpha",
            "hostname": "alpha",
            "process_id": "process:worker",
            "process_name": "worker",
        },
    )
    return run_perception_adapter_batch(
        generic_endpoint_process_adapter(),
        [event],
        batch_id="batch:endpoint",
    ).perception_batch


def _iam_batch() -> PerceptionBatch:
    event = SanitizedTelemetryEvent(
        event_id="event:iam",
        source_type=TelemetrySourceType.CLOUD_IAM,
        source_instance="iam:one",
        payload_sha256="b" * 64,
        fields={
            "identity_id": "identity:alice",
            "principal": "alice",
            "resource_id": "cloud:prod",
            "resource_type": "cluster",
        },
    )
    return run_perception_adapter_batch(
        generic_cloud_iam_adapter(),
        [event],
        batch_id="batch:iam",
    ).perception_batch


def test_fusion_preserves_multiple_sensor_provenance_without_inflating_status() -> None:
    result = fuse_perception_batches(
        [_endpoint_batch(), _iam_batch()],
        fused_batch_id="fusion:test",
    )
    graph = compile_perception_batch(result.perception_batch, graph_id="incident:test")

    assert len(result.receipt.source_principals) == 2
    assert result.receipt.observation_count == 2
    assert all(relation.status is EvidenceStatus.OBSERVED for relation in graph.relations)
    evidence_sources = {
        evidence.source for relation in graph.relations for evidence in relation.evidence
    }
    assert evidence_sources == {"ENDPOINT:edr:one", "CLOUD_IAM:iam:one"}


def test_fusion_rejects_duplicate_observation_replay_across_batches() -> None:
    original = _endpoint_batch()
    replay = PerceptionBatch(
        batch_id="batch:replay",
        observations=[original.observations[0].model_copy(deep=True)],
    )
    with pytest.raises(ValueError, match="replay/duplicate observation_id"):
        fuse_perception_batches([original, replay], fused_batch_id="fusion:replay")


def test_fusion_rejects_evidence_id_reused_with_different_digest() -> None:
    first = PerceptionObservation(
        observation_id="obs:first",
        source_type=TelemetrySourceType.ENDPOINT,
        source_instance="edr:one",
        source_entity=PerceivedEntity(
            entity_id="device:alpha",
            entity_type=CyberEntityType.DEVICE,
        ),
        target_entity=PerceivedEntity(
            entity_id="process:one",
            entity_type=CyberEntityType.PROCESS,
        ),
        relation_type="EXECUTED_PROCESS",
        confidence=1.0,
        evidence_id="evidence:shared",
        evidence_sha256="a" * 64,
    )
    second = PerceptionObservation(
        observation_id="obs:second",
        source_type=TelemetrySourceType.CLOUD_IAM,
        source_instance="iam:one",
        source_entity=PerceivedEntity(
            entity_id="identity:alice",
            entity_type=CyberEntityType.IDENTITY,
        ),
        target_entity=PerceivedEntity(
            entity_id="cloud:prod",
            entity_type=CyberEntityType.CLOUD_RESOURCE,
        ),
        relation_type="ACCESSED_CLOUD_RESOURCE",
        confidence=1.0,
        evidence_id="evidence:shared",
        evidence_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="different digest"):
        fuse_perception_batches(
            [
                PerceptionBatch(batch_id="batch:first", observations=[first]),
                PerceptionBatch(batch_id="batch:second", observations=[second]),
            ],
            fused_batch_id="fusion:evidence-conflict",
        )


def test_fusion_rejects_cross_sensor_entity_label_conflict() -> None:
    first = PerceptionObservation(
        observation_id="obs:label-one",
        source_type=TelemetrySourceType.ENDPOINT,
        source_instance="edr:one",
        source_entity=PerceivedEntity(
            entity_id="device:alpha",
            entity_type=CyberEntityType.DEVICE,
            labels={"hostname": "alpha"},
        ),
        target_entity=PerceivedEntity(
            entity_id="process:one",
            entity_type=CyberEntityType.PROCESS,
        ),
        relation_type="EXECUTED_PROCESS",
        confidence=1.0,
        evidence_id="evidence:one",
        evidence_sha256="a" * 64,
    )
    second = first.model_copy(
        deep=True,
        update={
            "observation_id": "obs:label-two",
            "source_instance": "edr:two",
            "source_entity": PerceivedEntity(
                entity_id="device:alpha",
                entity_type=CyberEntityType.DEVICE,
                labels={"hostname": "not-alpha"},
            ),
            "evidence_id": "evidence:two",
            "evidence_sha256": "b" * 64,
        },
    )
    with pytest.raises(ValueError, match="entity label conflict"):
        fuse_perception_batches(
            [
                PerceptionBatch(batch_id="batch:one", observations=[first]),
                PerceptionBatch(batch_id="batch:two", observations=[second]),
            ],
            fused_batch_id="fusion:label-conflict",
        )


def test_fusion_digest_is_stable_across_batch_input_order() -> None:
    first = fuse_perception_batches(
        [_endpoint_batch(), _iam_batch()],
        fused_batch_id="fusion:stable",
    )
    second = fuse_perception_batches(
        [_iam_batch(), _endpoint_batch()],
        fused_batch_id="fusion:stable",
    )

    assert first.receipt.fused_batch_sha256 == second.receipt.fused_batch_sha256
    assert first.receipt.model_dump() == second.receipt.model_dump()
