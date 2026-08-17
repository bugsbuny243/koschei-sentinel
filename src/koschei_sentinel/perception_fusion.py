from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_perception import PerceivedEntity, PerceptionBatch, PerceptionObservation
from koschei_sentinel.models import StrictModel

_DIGEST = r"^[a-f0-9]{64}$"


class FusionInputRef(StrictModel):
    batch_id: str
    batch_sha256: str = Field(pattern=_DIGEST)


class PerceptionFusionReceipt(StrictModel):
    schema_version: Literal["sentinel.perception-fusion-receipt.v1"] = (
        "sentinel.perception-fusion-receipt.v1"
    )
    fused_batch_id: str
    input_batches: list[FusionInputRef]
    observation_count: int = Field(ge=1)
    unique_evidence_count: int = Field(ge=1)
    source_principals: list[str]
    source_types: list[str]
    observations_per_source: dict[str, int]
    fused_batch_sha256: str = Field(pattern=_DIGEST)


class PerceptionFusionResult(StrictModel):
    schema_version: Literal["sentinel.perception-fusion-result.v1"] = (
        "sentinel.perception-fusion-result.v1"
    )
    perception_batch: PerceptionBatch
    receipt: PerceptionFusionReceipt


def perception_batch_sha256(batch: PerceptionBatch) -> str:
    payload = json.dumps(
        batch.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_entity_consistency(
    known: dict[str, PerceivedEntity],
    entity: PerceivedEntity,
) -> None:
    previous = known.get(entity.entity_id)
    if previous is None:
        known[entity.entity_id] = entity.model_copy(deep=True)
        return
    if previous.entity_type is not entity.entity_type:
        raise ValueError(f"fusion entity type conflict for {entity.entity_id}")
    merged = dict(previous.labels)
    for key, value in entity.labels.items():
        existing = merged.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"fusion entity label conflict for {entity.entity_id}: {key}")
        merged[key] = value
    known[entity.entity_id] = previous.model_copy(update={"labels": merged})


def fuse_perception_batches(
    batches: list[PerceptionBatch],
    *,
    fused_batch_id: str,
) -> PerceptionFusionResult:
    if not batches:
        raise ValueError("perception fusion requires at least one input batch")
    batch_ids = [batch.batch_id for batch in batches]
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("perception fusion input batch_id values must be unique")

    observations: list[PerceptionObservation] = []
    observation_ids: set[str] = set()
    evidence_sha_by_id: dict[str, str] = {}
    entities: dict[str, PerceivedEntity] = {}
    observations_per_source: dict[str, int] = {}
    input_refs: list[FusionInputRef] = []

    for batch in sorted(batches, key=lambda row: row.batch_id):
        input_refs.append(
            FusionInputRef(
                batch_id=batch.batch_id,
                batch_sha256=perception_batch_sha256(batch),
            )
        )
        for observation in sorted(batch.observations, key=lambda row: row.observation_id):
            if observation.observation_id in observation_ids:
                raise ValueError(
                    f"perception fusion replay/duplicate observation_id: {observation.observation_id}"
                )
            observation_ids.add(observation.observation_id)

            previous_sha = evidence_sha_by_id.get(observation.evidence_id)
            if previous_sha is not None and previous_sha != observation.evidence_sha256:
                raise ValueError(
                    "perception fusion evidence_id reused with different digest: "
                    f"{observation.evidence_id}"
                )
            evidence_sha_by_id[observation.evidence_id] = observation.evidence_sha256

            _validate_entity_consistency(entities, observation.source_entity)
            _validate_entity_consistency(entities, observation.target_entity)

            principal = f"{observation.source_type.value}:{observation.source_instance}"
            observations_per_source[principal] = observations_per_source.get(principal, 0) + 1
            observations.append(observation)

    fused = PerceptionBatch(
        batch_id=fused_batch_id,
        observations=sorted(observations, key=lambda row: row.observation_id),
    )
    receipt = PerceptionFusionReceipt(
        fused_batch_id=fused_batch_id,
        input_batches=input_refs,
        observation_count=len(fused.observations),
        unique_evidence_count=len(evidence_sha_by_id),
        source_principals=sorted(observations_per_source),
        source_types=sorted({row.source_type.value for row in fused.observations}),
        observations_per_source=dict(sorted(observations_per_source.items())),
        fused_batch_sha256=perception_batch_sha256(fused),
    )
    return PerceptionFusionResult(perception_batch=fused, receipt=receipt)
