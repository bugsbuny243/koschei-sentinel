from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_state_graph import (
    CyberEntity,
    CyberEntityType,
    CyberRelation,
    CyberStateGraph,
    EvidenceRef,
    EvidenceStatus,
)
from koschei_sentinel.models import StrictModel


class TelemetrySourceType(StrEnum):
    IDENTITY = "IDENTITY"
    ENDPOINT = "ENDPOINT"
    PROCESS = "PROCESS"
    NETWORK = "NETWORK"
    CLOUD_IAM = "CLOUD_IAM"
    REPOSITORY = "REPOSITORY"
    CICD = "CICD"
    SIGNER_WALLET = "SIGNER_WALLET"
    BLOCKCHAIN = "BLOCKCHAIN"
    SECURITY_CONTROL = "SECURITY_CONTROL"


_SENSITIVE_LABEL_TOKENS = {
    "password",
    "passwd",
    "secret",
    "private_key",
    "private-key",
    "mnemonic",
    "seed_phrase",
    "seed-phrase",
    "api_key",
    "api-key",
    "access_token",
    "refresh_token",
    "bearer",
    "authorization",
    "session_cookie",
}


class PerceivedEntity(StrictModel):
    entity_id: str = Field(min_length=3, max_length=256)
    entity_type: CyberEntityType
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_sensitive_label_names(self) -> PerceivedEntity:
        for key in self.labels:
            normalized = key.strip().lower().replace(" ", "_")
            if normalized in _SENSITIVE_LABEL_TOKENS or any(
                token in normalized for token in ("password", "private_key", "mnemonic", "seed_phrase")
            ):
                raise ValueError(f"sensitive material must not be stored in entity labels: {key}")
        return self


class PerceptionObservation(StrictModel):
    observation_id: str = Field(min_length=3, max_length=256)
    source_type: TelemetrySourceType
    source_instance: str = Field(min_length=2, max_length=256)
    source_entity: PerceivedEntity
    target_entity: PerceivedEntity
    relation_type: str = Field(min_length=2, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_id: str = Field(min_length=3, max_length=256)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    observed_at: str | None = None


class PerceptionBatch(StrictModel):
    schema_version: Literal["sentinel.perception-batch.v1"] = "sentinel.perception-batch.v1"
    batch_id: str = Field(min_length=3, max_length=256)
    observations: list[PerceptionObservation] = Field(min_length=1, max_length=100000)

    @model_validator(mode="after")
    def observation_ids_are_unique(self) -> PerceptionBatch:
        ids = [row.observation_id for row in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("observation_id values must be unique inside a perception batch")
        return self


def _relation_id(source: str, relation_type: str, target: str) -> str:
    payload = f"{source}|{relation_type.strip().lower()}|{target}"
    return "observed:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def compile_perception_batch(batch: PerceptionBatch, *, graph_id: str) -> CyberStateGraph:
    entity_map: dict[str, CyberEntity] = {}
    relation_map: dict[str, CyberRelation] = {}

    def merge_entity(entity: PerceivedEntity) -> None:
        existing = entity_map.get(entity.entity_id)
        if existing is None:
            entity_map[entity.entity_id] = CyberEntity(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                labels=dict(entity.labels),
            )
            return
        if existing.entity_type is not entity.entity_type:
            raise ValueError(f"conflicting entity type for {entity.entity_id}")
        merged = dict(existing.labels)
        for key, value in entity.labels.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"conflicting entity label for {entity.entity_id}: {key}"
                )
            merged[key] = value
        existing.labels = merged

    for observation in batch.observations:
        merge_entity(observation.source_entity)
        merge_entity(observation.target_entity)
        relation_id = _relation_id(
            observation.source_entity.entity_id,
            observation.relation_type,
            observation.target_entity.entity_id,
        )
        evidence = EvidenceRef(
            evidence_id=observation.evidence_id,
            source=f"{observation.source_type.value}:{observation.source_instance}",
            content_sha256=observation.evidence_sha256,
            timestamp=observation.observed_at,
        )
        existing_relation = relation_map.get(relation_id)
        if existing_relation is None:
            relation_map[relation_id] = CyberRelation(
                relation_id=relation_id,
                source_entity_id=observation.source_entity.entity_id,
                target_entity_id=observation.target_entity.entity_id,
                relation_type=observation.relation_type,
                status=EvidenceStatus.OBSERVED,
                confidence=observation.confidence,
                evidence=[evidence],
                rationale="normalized defensive telemetry observation",
            )
            continue

        evidence_by_id = {item.evidence_id: item for item in existing_relation.evidence}
        previous = evidence_by_id.get(evidence.evidence_id)
        if previous is not None and previous.content_sha256 != evidence.content_sha256:
            raise ValueError(
                f"evidence_id reused with different content digest: {evidence.evidence_id}"
            )
        evidence_by_id[evidence.evidence_id] = evidence
        existing_relation.evidence = sorted(
            evidence_by_id.values(),
            key=lambda item: item.evidence_id,
        )
        existing_relation.confidence = max(existing_relation.confidence, observation.confidence)

    return CyberStateGraph(
        graph_id=graph_id,
        entities=sorted(entity_map.values(), key=lambda row: row.entity_id),
        relations=sorted(relation_map.values(), key=lambda row: row.relation_id),
    )
