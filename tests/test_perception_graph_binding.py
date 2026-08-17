import pytest

from koschei_sentinel.cyber_perception import (
    PerceivedEntity,
    PerceptionBatch,
    PerceptionObservation,
    TelemetrySourceType,
)
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.perception_graph_binding import (
    compile_bound_perception_graph,
    verify_perception_graph_receipt,
)


def _batch(batch_id: str = "batch:test") -> PerceptionBatch:
    return PerceptionBatch(
        batch_id=batch_id,
        observations=[
            PerceptionObservation(
                observation_id="obs:test",
                source_type=TelemetrySourceType.ENDPOINT,
                source_instance="edr:test",
                source_entity=PerceivedEntity(
                    entity_id="device:test",
                    entity_type=CyberEntityType.DEVICE,
                ),
                target_entity=PerceivedEntity(
                    entity_id="process:test",
                    entity_type=CyberEntityType.PROCESS,
                ),
                relation_type="EXECUTED_PROCESS",
                confidence=1.0,
                evidence_id="evidence:test",
                evidence_sha256="a" * 64,
            )
        ],
    )


def test_bound_graph_receipt_links_batch_and_graph_digests() -> None:
    bound = compile_bound_perception_graph(_batch(), graph_id="incident:test")

    assert bound.receipt.source_batch_id == "batch:test"
    assert len(bound.receipt.source_batch_sha256) == 64
    assert len(bound.receipt.graph_sha256) == 64
    verify_perception_graph_receipt(bound.graph, bound.receipt)


def test_tampered_graph_fails_receipt_verification() -> None:
    bound = compile_bound_perception_graph(_batch(), graph_id="incident:test")
    tampered = bound.graph.model_copy(deep=True)
    tampered.entities[0].labels["hostname"] = "changed"

    with pytest.raises(ValueError, match="graph digest mismatch"):
        verify_perception_graph_receipt(tampered, bound.receipt)


def test_tampered_receipt_integrity_digest_is_rejected() -> None:
    bound = compile_bound_perception_graph(_batch(), graph_id="incident:test")
    receipt = bound.receipt.model_copy(update={"receipt_sha256": "f" * 64})

    with pytest.raises(ValueError, match="integrity digest"):
        verify_perception_graph_receipt(bound.graph, receipt)
