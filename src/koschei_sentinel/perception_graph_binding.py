from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_perception import PerceptionBatch, compile_perception_batch
from koschei_sentinel.cyber_state_graph import CyberStateGraph
from koschei_sentinel.models import StrictModel
from koschei_sentinel.perception_fusion import perception_batch_sha256

_DIGEST = r"^[a-f0-9]{64}$"


class PerceptionGraphReceipt(StrictModel):
    schema_version: Literal["sentinel.perception-graph-receipt.v1"] = (
        "sentinel.perception-graph-receipt.v1"
    )
    graph_id: str
    source_batch_id: str
    source_batch_sha256: str = Field(pattern=_DIGEST)
    graph_sha256: str = Field(pattern=_DIGEST)
    entities: int = Field(ge=0)
    relations: int = Field(ge=0)
    receipt_sha256: str = Field(pattern=_DIGEST)


class BoundPerceptionGraph(StrictModel):
    schema_version: Literal["sentinel.bound-perception-graph.v1"] = (
        "sentinel.bound-perception-graph.v1"
    )
    graph: CyberStateGraph
    receipt: PerceptionGraphReceipt

    @model_validator(mode="after")
    def binding_is_self_consistent(self) -> BoundPerceptionGraph:
        if self.graph.graph_id != self.receipt.graph_id:
            raise ValueError("bound perception graph_id does not match receipt")
        if cyber_state_graph_sha256(self.graph) != self.receipt.graph_sha256:
            raise ValueError("bound perception graph digest does not match receipt")
        return self


def cyber_state_graph_sha256(graph: CyberStateGraph) -> str:
    payload = json.dumps(
        graph.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _receipt_digest(
    *,
    graph_id: str,
    source_batch_id: str,
    source_batch_sha256: str,
    graph_sha256: str,
    entities: int,
    relations: int,
) -> str:
    payload = "|".join(
        [
            graph_id,
            source_batch_id,
            source_batch_sha256,
            graph_sha256,
            str(entities),
            str(relations),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_bound_perception_graph(
    batch: PerceptionBatch,
    *,
    graph_id: str,
) -> BoundPerceptionGraph:
    graph = compile_perception_batch(batch, graph_id=graph_id)
    batch_sha = perception_batch_sha256(batch)
    graph_sha = cyber_state_graph_sha256(graph)
    receipt_sha = _receipt_digest(
        graph_id=graph.graph_id,
        source_batch_id=batch.batch_id,
        source_batch_sha256=batch_sha,
        graph_sha256=graph_sha,
        entities=len(graph.entities),
        relations=len(graph.relations),
    )
    receipt = PerceptionGraphReceipt(
        graph_id=graph.graph_id,
        source_batch_id=batch.batch_id,
        source_batch_sha256=batch_sha,
        graph_sha256=graph_sha,
        entities=len(graph.entities),
        relations=len(graph.relations),
        receipt_sha256=receipt_sha,
    )
    return BoundPerceptionGraph(graph=graph, receipt=receipt)


def verify_perception_graph_receipt(
    graph: CyberStateGraph,
    receipt: PerceptionGraphReceipt,
) -> None:
    if graph.graph_id != receipt.graph_id:
        raise ValueError("perception graph receipt references a different graph_id")
    actual_graph_sha = cyber_state_graph_sha256(graph)
    if actual_graph_sha != receipt.graph_sha256:
        raise ValueError("perception graph receipt graph digest mismatch")
    expected_receipt_sha = _receipt_digest(
        graph_id=receipt.graph_id,
        source_batch_id=receipt.source_batch_id,
        source_batch_sha256=receipt.source_batch_sha256,
        graph_sha256=receipt.graph_sha256,
        entities=receipt.entities,
        relations=receipt.relations,
    )
    if expected_receipt_sha != receipt.receipt_sha256:
        raise ValueError("perception graph receipt integrity digest mismatch")
    if receipt.entities != len(graph.entities) or receipt.relations != len(graph.relations):
        raise ValueError("perception graph receipt counts do not match graph")
