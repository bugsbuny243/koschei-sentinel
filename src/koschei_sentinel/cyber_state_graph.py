from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class EvidenceStatus(StrEnum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    PREDICTED = "PREDICTED"
    DISPROVED = "DISPROVED"


class CyberEntityType(StrEnum):
    IDENTITY = "IDENTITY"
    DEVICE = "DEVICE"
    PROCESS = "PROCESS"
    CREDENTIAL = "CREDENTIAL"
    REPOSITORY = "REPOSITORY"
    PIPELINE = "PIPELINE"
    ARTIFACT = "ARTIFACT"
    CLOUD_RESOURCE = "CLOUD_RESOURCE"
    WALLET = "WALLET"
    TRANSACTION = "TRANSACTION"
    PROTOCOL = "PROTOCOL"
    NETWORK_ENDPOINT = "NETWORK_ENDPOINT"
    THREAT_ACTOR = "THREAT_ACTOR"
    MALWARE = "MALWARE"


class CyberEntity(StrictModel):
    entity_id: str = Field(min_length=3, max_length=256)
    entity_type: CyberEntityType
    labels: dict[str, str] = Field(default_factory=dict)


class EvidenceRef(StrictModel):
    evidence_id: str = Field(min_length=3, max_length=256)
    source: str = Field(min_length=2, max_length=128)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    timestamp: str | None = None


class CyberRelation(StrictModel):
    relation_id: str = Field(min_length=3, max_length=256)
    source_entity_id: str
    target_entity_id: str
    relation_type: str = Field(min_length=2, max_length=128)
    status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str | None = None

    @model_validator(mode="after")
    def evidence_rules(self) -> CyberRelation:
        if self.status is EvidenceStatus.OBSERVED and not self.evidence:
            raise ValueError("OBSERVED relations require evidence")
        if self.status is EvidenceStatus.DISPROVED and self.confidence > 0.5:
            raise ValueError("DISPROVED relations cannot retain high confidence")
        return self


class CyberStateGraph(StrictModel):
    schema_version: Literal["sentinel.cyber-state-graph.v1"] = "sentinel.cyber-state-graph.v1"
    graph_id: str = Field(min_length=3, max_length=256)
    entities: list[CyberEntity] = Field(default_factory=list)
    relations: list[CyberRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def graph_integrity(self) -> CyberStateGraph:
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity_id values must be unique")
        known = set(ids)
        relation_ids: set[str] = set()
        for relation in self.relations:
            if relation.relation_id in relation_ids:
                raise ValueError("relation_id values must be unique")
            relation_ids.add(relation.relation_id)
            if relation.source_entity_id not in known or relation.target_entity_id not in known:
                raise ValueError("relations must reference known entities")
        return self
