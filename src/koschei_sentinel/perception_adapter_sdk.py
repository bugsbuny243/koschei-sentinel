from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from pydantic import Field, model_validator

from koschei_sentinel.cyber_perception import (
    PerceivedEntity,
    PerceptionBatch,
    PerceptionObservation,
    TelemetrySourceType,
)
from koschei_sentinel.cyber_state_graph import CyberEntityType
from koschei_sentinel.models import StrictModel

Scalar: TypeAlias = str | int | float | bool | None
_DIGEST = r"^[a-f0-9]{64}$"
_FORBIDDEN_FIELD_TOKENS = {
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
    "authorization",
    "session_cookie",
}


class PerceptionAdapterDescriptor(StrictModel):
    schema_version: Literal["sentinel.perception-adapter-descriptor.v1"] = (
        "sentinel.perception-adapter-descriptor.v1"
    )
    adapter_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    vendor: str = Field(min_length=2, max_length=128)
    product: str = Field(min_length=2, max_length=128)
    adapter_version: str = Field(min_length=1, max_length=64)
    supported_source_types: list[TelemetrySourceType] = Field(min_length=1, max_length=32)
    deterministic: Literal[True] = True
    network_access: Literal[False] = False
    secret_handling: Literal["REJECT"] = "REJECT"

    @model_validator(mode="after")
    def source_types_are_unique(self) -> PerceptionAdapterDescriptor:
        if len(self.supported_source_types) != len(set(self.supported_source_types)):
            raise ValueError("perception adapter supported_source_types must be unique")
        return self


class SanitizedTelemetryEvent(StrictModel):
    schema_version: Literal["sentinel.sanitized-telemetry-event.v1"] = (
        "sentinel.sanitized-telemetry-event.v1"
    )
    event_id: str = Field(min_length=3, max_length=256)
    source_type: TelemetrySourceType
    source_instance: str = Field(min_length=2, max_length=256)
    payload_sha256: str = Field(pattern=_DIGEST)
    observed_at: str | None = None
    fields: dict[str, Scalar] = Field(default_factory=dict, max_length=512)

    @model_validator(mode="after")
    def reject_sensitive_field_names(self) -> SanitizedTelemetryEvent:
        for key in self.fields:
            normalized = key.strip().lower().replace(" ", "_")
            if normalized in _FORBIDDEN_FIELD_TOKENS or any(
                token in normalized
                for token in ("password", "private_key", "mnemonic", "seed_phrase")
            ):
                raise ValueError(f"sensitive field is forbidden in sanitized telemetry: {key}")
        return self


class PerceptionAdapter(Protocol):
    descriptor: PerceptionAdapterDescriptor

    def normalize(self, event: SanitizedTelemetryEvent) -> list[PerceptionObservation]: ...


class EntityMappingSpec(StrictModel):
    id_field: str = Field(min_length=1, max_length=128)
    entity_type: CyberEntityType
    label_fields: dict[str, str] = Field(default_factory=dict, max_length=64)


class DeclarativeAdapterSpec(StrictModel):
    schema_version: Literal["sentinel.declarative-perception-adapter.v1"] = (
        "sentinel.declarative-perception-adapter.v1"
    )
    descriptor: PerceptionAdapterDescriptor
    source: EntityMappingSpec
    target: EntityMappingSpec
    relation_type: str = Field(min_length=2, max_length=128)
    default_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_field: str | None = Field(default=None, max_length=128)


class AdapterRunReceipt(StrictModel):
    schema_version: Literal["sentinel.perception-adapter-receipt.v1"] = (
        "sentinel.perception-adapter-receipt.v1"
    )
    adapter_id: str
    event_id: str
    payload_sha256: str = Field(pattern=_DIGEST)
    observation_count: int = Field(ge=0)
    observation_ids: list[str]
    output_sha256: str = Field(pattern=_DIGEST)


class AdapterBatchResult(StrictModel):
    schema_version: Literal["sentinel.perception-adapter-batch-result.v1"] = (
        "sentinel.perception-adapter-batch-result.v1"
    )
    perception_batch: PerceptionBatch
    receipts: list[AdapterRunReceipt]
    batch_sha256: str = Field(pattern=_DIGEST)


@dataclass(frozen=True)
class DeclarativePerceptionAdapter:
    spec: DeclarativeAdapterSpec

    @property
    def descriptor(self) -> PerceptionAdapterDescriptor:
        return self.spec.descriptor

    def _required_string(self, event: SanitizedTelemetryEvent, field: str) -> str:
        value = event.fields.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"required adapter field is missing or not a string: {field}")
        return value.strip()

    def _entity(self, event: SanitizedTelemetryEvent, mapping: EntityMappingSpec) -> PerceivedEntity:
        labels: dict[str, str] = {}
        for label, field in mapping.label_fields.items():
            value = event.fields.get(field)
            if value is not None:
                labels[label] = str(value)
        return PerceivedEntity(
            entity_id=self._required_string(event, mapping.id_field),
            entity_type=mapping.entity_type,
            labels=labels,
        )

    def _confidence(self, event: SanitizedTelemetryEvent) -> float:
        if self.spec.confidence_field is None:
            return self.spec.default_confidence
        value = event.fields.get(self.spec.confidence_field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("configured confidence_field must contain a numeric value")
        return float(value)

    def normalize(self, event: SanitizedTelemetryEvent) -> list[PerceptionObservation]:
        source = self._entity(event, self.spec.source)
        target = self._entity(event, self.spec.target)
        observation_id = deterministic_observation_id(
            adapter_id=self.descriptor.adapter_id,
            event_id=event.event_id,
            source_entity_id=source.entity_id,
            relation_type=self.spec.relation_type,
            target_entity_id=target.entity_id,
        )
        evidence_id = deterministic_evidence_id(
            adapter_id=self.descriptor.adapter_id,
            event_id=event.event_id,
            payload_sha256=event.payload_sha256,
        )
        return [
            PerceptionObservation(
                observation_id=observation_id,
                source_type=event.source_type,
                source_instance=event.source_instance,
                source_entity=source,
                target_entity=target,
                relation_type=self.spec.relation_type,
                confidence=self._confidence(event),
                evidence_id=evidence_id,
                evidence_sha256=event.payload_sha256,
                observed_at=event.observed_at,
            )
        ]


def deterministic_observation_id(
    *,
    adapter_id: str,
    event_id: str,
    source_entity_id: str,
    relation_type: str,
    target_entity_id: str,
) -> str:
    payload = "|".join(
        [adapter_id, event_id, source_entity_id, relation_type.strip().lower(), target_entity_id]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"obs:{adapter_id}:{digest}"


def deterministic_evidence_id(*, adapter_id: str, event_id: str, payload_sha256: str) -> str:
    digest = hashlib.sha256(
        f"{adapter_id}|{event_id}|{payload_sha256}".encode()
    ).hexdigest()[:32]
    return f"evidence:{adapter_id}:{digest}"


def _observations_digest(rows: list[PerceptionObservation]) -> str:
    encoded = json.dumps(
        [row.model_dump(mode="json") for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_perception_adapter(
    adapter: PerceptionAdapter,
    event: SanitizedTelemetryEvent,
) -> tuple[list[PerceptionObservation], AdapterRunReceipt]:
    descriptor = adapter.descriptor
    if event.source_type not in descriptor.supported_source_types:
        raise ValueError("telemetry source type is not supported by this adapter")

    rows = adapter.normalize(event)
    ids = [row.observation_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("adapter emitted duplicate observation_id values")

    observation_prefix = f"obs:{descriptor.adapter_id}:"
    evidence_prefix = f"evidence:{descriptor.adapter_id}:"
    for row in rows:
        if row.source_type is not event.source_type:
            raise ValueError("adapter attempted to change telemetry source type")
        if row.source_instance != event.source_instance:
            raise ValueError("adapter attempted to change telemetry source instance")
        if row.evidence_sha256 != event.payload_sha256:
            raise ValueError("adapter evidence digest does not match telemetry payload digest")
        if not row.observation_id.startswith(observation_prefix):
            raise ValueError("adapter observation_id is outside its namespace")
        if not row.evidence_id.startswith(evidence_prefix):
            raise ValueError("adapter evidence_id is outside its namespace")

    ordered = sorted(rows, key=lambda row: row.observation_id)
    receipt = AdapterRunReceipt(
        adapter_id=descriptor.adapter_id,
        event_id=event.event_id,
        payload_sha256=event.payload_sha256,
        observation_count=len(ordered),
        observation_ids=[row.observation_id for row in ordered],
        output_sha256=_observations_digest(ordered),
    )
    return ordered, receipt


def run_perception_adapter_batch(
    adapter: PerceptionAdapter,
    events: list[SanitizedTelemetryEvent],
    *,
    batch_id: str,
) -> AdapterBatchResult:
    if not events:
        raise ValueError("perception adapter batch requires at least one telemetry event")
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("telemetry event_id values must be unique inside an adapter batch")

    observations: list[PerceptionObservation] = []
    receipts: list[AdapterRunReceipt] = []
    for event in sorted(events, key=lambda row: row.event_id):
        rows, receipt = run_perception_adapter(adapter, event)
        observations.extend(rows)
        receipts.append(receipt)

    batch = PerceptionBatch(batch_id=batch_id, observations=observations)
    encoded = json.dumps(
        {
            "perception_batch": batch.model_dump(mode="json"),
            "receipts": [row.model_dump(mode="json") for row in receipts],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return AdapterBatchResult(
        perception_batch=batch,
        receipts=receipts,
        batch_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )
