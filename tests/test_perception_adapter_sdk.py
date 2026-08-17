import pytest

from koschei_sentinel.cyber_perception import (
    PerceivedEntity,
    PerceptionObservation,
    TelemetrySourceType,
    compile_perception_batch,
)
from koschei_sentinel.cyber_state_graph import CyberEntityType, EvidenceStatus
from koschei_sentinel.perception_adapter_profiles import generic_endpoint_process_adapter
from koschei_sentinel.perception_adapter_sdk import (
    PerceptionAdapterDescriptor,
    SanitizedTelemetryEvent,
    run_perception_adapter,
    run_perception_adapter_batch,
)


def _endpoint_event(event_id: str = "event:1") -> SanitizedTelemetryEvent:
    return SanitizedTelemetryEvent(
        event_id=event_id,
        source_type=TelemetrySourceType.ENDPOINT,
        source_instance="edr:test",
        payload_sha256="a" * 64,
        observed_at="2026-08-17T20:00:00+03:00",
        fields={
            "device_id": "device:alpha",
            "hostname": "alpha",
            "platform": "linux",
            "process_id": f"process:{event_id}",
            "process_name": "worker",
            "image_sha256": "b" * 64,
        },
    )


def test_reference_adapter_produces_observed_graph_evidence() -> None:
    result = run_perception_adapter_batch(
        generic_endpoint_process_adapter(),
        [_endpoint_event()],
        batch_id="adapter-batch:test",
    )
    graph = compile_perception_batch(result.perception_batch, graph_id="incident:test")

    assert len(graph.relations) == 1
    assert graph.relations[0].status is EvidenceStatus.OBSERVED
    assert graph.relations[0].relation_type == "executes"
    assert result.receipts[0].payload_sha256 == "a" * 64


def test_sanitized_event_rejects_sensitive_fields_before_adapter_execution() -> None:
    with pytest.raises(ValueError, match="sensitive field"):
        SanitizedTelemetryEvent(
            event_id="event:secret",
            source_type=TelemetrySourceType.ENDPOINT,
            source_instance="edr:test",
            payload_sha256="a" * 64,
            fields={"private_key": "must-never-enter-perception"},
        )


class _BadSourceTypeAdapter:
    descriptor = PerceptionAdapterDescriptor(
        adapter_id="test.bad-source",
        vendor="test",
        product="bad-source",
        adapter_version="1",
        supported_source_types=[TelemetrySourceType.ENDPOINT],
    )

    def normalize(self, event: SanitizedTelemetryEvent) -> list[PerceptionObservation]:
        return [
            PerceptionObservation(
                observation_id="obs:test.bad-source:1234567890abcdef",
                source_type=TelemetrySourceType.CLOUD_IAM,
                source_instance=event.source_instance,
                source_entity=PerceivedEntity(
                    entity_id="device:alpha",
                    entity_type=CyberEntityType.DEVICE,
                ),
                target_entity=PerceivedEntity(
                    entity_id="process:alpha",
                    entity_type=CyberEntityType.PROCESS,
                ),
                relation_type="executes",
                confidence=1.0,
                evidence_id="evidence:test.bad-source:1234567890abcdef",
                evidence_sha256=event.payload_sha256,
            )
        ]


def test_adapter_cannot_change_telemetry_source_type() -> None:
    with pytest.raises(ValueError, match="change telemetry source type"):
        run_perception_adapter(_BadSourceTypeAdapter(), _endpoint_event())


class _BadEvidenceAdapter:
    descriptor = PerceptionAdapterDescriptor(
        adapter_id="test.bad-evidence",
        vendor="test",
        product="bad-evidence",
        adapter_version="1",
        supported_source_types=[TelemetrySourceType.ENDPOINT],
    )

    def normalize(self, event: SanitizedTelemetryEvent) -> list[PerceptionObservation]:
        return [
            PerceptionObservation(
                observation_id="obs:test.bad-evidence:1234567890abcdef",
                source_type=event.source_type,
                source_instance=event.source_instance,
                source_entity=PerceivedEntity(
                    entity_id="device:alpha",
                    entity_type=CyberEntityType.DEVICE,
                ),
                target_entity=PerceivedEntity(
                    entity_id="process:alpha",
                    entity_type=CyberEntityType.PROCESS,
                ),
                relation_type="executes",
                confidence=1.0,
                evidence_id="evidence:test.bad-evidence:1234567890abcdef",
                evidence_sha256="f" * 64,
            )
        ]


def test_adapter_cannot_spoof_evidence_digest() -> None:
    with pytest.raises(ValueError, match="evidence digest"):
        run_perception_adapter(_BadEvidenceAdapter(), _endpoint_event())


class _BadNamespaceAdapter:
    descriptor = PerceptionAdapterDescriptor(
        adapter_id="test.namespace",
        vendor="test",
        product="namespace",
        adapter_version="1",
        supported_source_types=[TelemetrySourceType.ENDPOINT],
    )

    def normalize(self, event: SanitizedTelemetryEvent) -> list[PerceptionObservation]:
        return [
            PerceptionObservation(
                observation_id="obs:other-adapter:1234567890abcdef",
                source_type=event.source_type,
                source_instance=event.source_instance,
                source_entity=PerceivedEntity(
                    entity_id="device:alpha",
                    entity_type=CyberEntityType.DEVICE,
                ),
                target_entity=PerceivedEntity(
                    entity_id="process:alpha",
                    entity_type=CyberEntityType.PROCESS,
                ),
                relation_type="executes",
                confidence=1.0,
                evidence_id="evidence:test.namespace:1234567890abcdef",
                evidence_sha256=event.payload_sha256,
            )
        ]


def test_adapter_cannot_write_outside_observation_namespace() -> None:
    with pytest.raises(ValueError, match="outside its namespace"):
        run_perception_adapter(_BadNamespaceAdapter(), _endpoint_event())


def test_adapter_batch_digest_is_deterministic_across_input_order() -> None:
    adapter = generic_endpoint_process_adapter()
    first = run_perception_adapter_batch(
        adapter,
        [_endpoint_event("event:2"), _endpoint_event("event:1")],
        batch_id="adapter-batch:stable",
    )
    second = run_perception_adapter_batch(
        adapter,
        [_endpoint_event("event:1"), _endpoint_event("event:2")],
        batch_id="adapter-batch:stable",
    )

    assert first.batch_sha256 == second.batch_sha256
    assert first.perception_batch.model_dump() == second.perception_batch.model_dump()
