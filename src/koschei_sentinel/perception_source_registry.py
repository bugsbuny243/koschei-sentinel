from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.cyber_perception import (
    PerceptionBatch,
    PerceptionObservation,
    TelemetrySourceType,
)
from koschei_sentinel.models import StrictModel
from koschei_sentinel.perception_adapter_sdk import AdapterBatchResult
from koschei_sentinel.perception_fusion import perception_batch_sha256

_DIGEST = r"^[a-f0-9]{64}$"


class PerceptionSourceEnrollment(StrictModel):
    schema_version: Literal["sentinel.perception-source-enrollment.v1"] = (
        "sentinel.perception-source-enrollment.v1"
    )
    source_type: TelemetrySourceType
    source_instance: str = Field(min_length=2, max_length=256)
    allowed_adapter_ids: list[str] = Field(min_length=1, max_length=64)
    independence_domain: str = Field(min_length=2, max_length=128)
    enabled: bool = True

    @property
    def principal(self) -> str:
        return f"{self.source_type.value}:{self.source_instance}"

    @model_validator(mode="after")
    def adapter_ids_are_unique(self) -> "PerceptionSourceEnrollment":
        if len(self.allowed_adapter_ids) != len(set(self.allowed_adapter_ids)):
            raise ValueError("source enrollment allowed_adapter_ids must be unique")
        return self


class PerceptionSourceRegistry(StrictModel):
    schema_version: Literal["sentinel.perception-source-registry.v1"] = (
        "sentinel.perception-source-registry.v1"
    )
    registry_id: str = Field(min_length=3, max_length=256)
    enrollments: list[PerceptionSourceEnrollment] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def principals_are_unique(self) -> "PerceptionSourceRegistry":
        principals = [row.principal for row in self.enrollments]
        if len(principals) != len(set(principals)):
            raise ValueError("perception source registry principals must be unique")
        return self


class PerceptionAdmissionReceipt(StrictModel):
    schema_version: Literal["sentinel.perception-admission-receipt.v1"] = (
        "sentinel.perception-admission-receipt.v1"
    )
    admission_id: str
    registry_id: str
    registry_sha256: str = Field(pattern=_DIGEST)
    adapter_id: str
    adapter_batch_sha256: str = Field(pattern=_DIGEST)
    perception_batch_sha256: str = Field(pattern=_DIGEST)
    source_principals: list[str]
    independence_domains: list[str]
    admitted: Literal[True] = True
    admission_sha256: str = Field(pattern=_DIGEST)


class AdmittedPerceptionBatch(StrictModel):
    schema_version: Literal["sentinel.admitted-perception-batch.v1"] = (
        "sentinel.admitted-perception-batch.v1"
    )
    adapter_result: AdapterBatchResult
    perception_batch: PerceptionBatch
    admission: PerceptionAdmissionReceipt

    @model_validator(mode="after")
    def envelope_integrity(self) -> "AdmittedPerceptionBatch":
        if (
            self.adapter_result.perception_batch.model_dump(mode="json")
            != self.perception_batch.model_dump(mode="json")
        ):
            raise ValueError("admitted perception batch differs from bound adapter result")
        if self.adapter_result.batch_sha256 != self.admission.adapter_batch_sha256:
            raise ValueError("admission adapter batch digest does not match bound adapter result")
        actual = perception_batch_sha256(self.perception_batch)
        if actual != self.admission.perception_batch_sha256:
            raise ValueError("admitted perception batch digest does not match admission receipt")
        return self


def perception_registry_sha256(registry: PerceptionSourceRegistry) -> str:
    payload = json.dumps(
        registry.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observations_digest(rows: list[PerceptionObservation]) -> str:
    payload = json.dumps(
        [row.model_dump(mode="json") for row in sorted(rows, key=lambda row: row.observation_id)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _adapter_batch_digest(result: AdapterBatchResult) -> str:
    payload = json.dumps(
        {
            "perception_batch": result.perception_batch.model_dump(mode="json"),
            "receipts": [row.model_dump(mode="json") for row in result.receipts],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_adapter_result(result: AdapterBatchResult) -> str:
    actual_batch_sha = _adapter_batch_digest(result)
    if actual_batch_sha != result.batch_sha256:
        raise ValueError("adapter batch result digest does not match its contents")
    if not result.receipts:
        raise ValueError("adapter batch result requires at least one receipt")

    observations = {
        row.observation_id: row for row in result.perception_batch.observations
    }
    claimed_observation_ids: set[str] = set()
    adapter_ids = {receipt.adapter_id for receipt in result.receipts}
    if len(adapter_ids) != 1:
        raise ValueError("adapter batch result must contain receipts from exactly one adapter")

    for receipt in result.receipts:
        rows: list[PerceptionObservation] = []
        if receipt.observation_count != len(receipt.observation_ids):
            raise ValueError("adapter receipt observation_count does not match observation_ids")
        for observation_id in receipt.observation_ids:
            if observation_id in claimed_observation_ids:
                raise ValueError("adapter receipt observation_id is claimed more than once")
            observation = observations.get(observation_id)
            if observation is None:
                raise ValueError("adapter receipt references an unknown observation_id")
            if observation.evidence_sha256 != receipt.payload_sha256:
                raise ValueError("adapter receipt payload digest does not match observation evidence")
            claimed_observation_ids.add(observation_id)
            rows.append(observation)
        if _observations_digest(rows) != receipt.output_sha256:
            raise ValueError("adapter receipt output digest does not match observations")

    if claimed_observation_ids != set(observations):
        raise ValueError("one or more perception observations are not bound to an adapter receipt")
    return next(iter(adapter_ids))


def _admission_digest(
    *,
    registry_sha256: str,
    adapter_id: str,
    adapter_batch_sha256: str,
    perception_batch_sha256: str,
    source_principals: list[str],
    independence_domains: list[str],
) -> str:
    payload = "|".join(
        [
            registry_sha256,
            adapter_id,
            adapter_batch_sha256,
            perception_batch_sha256,
            ",".join(source_principals),
            ",".join(independence_domains),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def admit_perception_adapter_result(
    result: AdapterBatchResult,
    registry: PerceptionSourceRegistry,
) -> AdmittedPerceptionBatch:
    adapter_id = _verify_adapter_result(result)
    enrollments = {row.principal: row for row in registry.enrollments}

    principals = sorted(
        {
            f"{observation.source_type.value}:{observation.source_instance}"
            for observation in result.perception_batch.observations
        }
    )
    domains: set[str] = set()
    for principal in principals:
        enrollment = enrollments.get(principal)
        if enrollment is None:
            raise ValueError(f"perception source is not enrolled: {principal}")
        if not enrollment.enabled:
            raise ValueError(f"perception source enrollment is disabled: {principal}")
        if adapter_id not in set(enrollment.allowed_adapter_ids):
            raise ValueError(
                f"adapter {adapter_id} is not permitted for perception source {principal}"
            )
        domains.add(enrollment.independence_domain)

    registry_sha = perception_registry_sha256(registry)
    perception_sha = perception_batch_sha256(result.perception_batch)
    digest = _admission_digest(
        registry_sha256=registry_sha,
        adapter_id=adapter_id,
        adapter_batch_sha256=result.batch_sha256,
        perception_batch_sha256=perception_sha,
        source_principals=principals,
        independence_domains=sorted(domains),
    )
    admission = PerceptionAdmissionReceipt(
        admission_id=f"admission:{result.perception_batch.batch_id}:{digest[:16]}",
        registry_id=registry.registry_id,
        registry_sha256=registry_sha,
        adapter_id=adapter_id,
        adapter_batch_sha256=result.batch_sha256,
        perception_batch_sha256=perception_sha,
        source_principals=principals,
        independence_domains=sorted(domains),
        admission_sha256=digest,
    )
    return AdmittedPerceptionBatch(
        adapter_result=result,
        perception_batch=result.perception_batch,
        admission=admission,
    )


def verify_admitted_perception_batch(
    admitted: AdmittedPerceptionBatch,
    registry: PerceptionSourceRegistry,
) -> AdmittedPerceptionBatch:
    expected = admit_perception_adapter_result(admitted.adapter_result, registry)
    if (
        expected.perception_batch.model_dump(mode="json")
        != admitted.perception_batch.model_dump(mode="json")
    ):
        raise ValueError("admitted perception batch does not match verified adapter result")
    if expected.admission.model_dump(mode="json") != admitted.admission.model_dump(mode="json"):
        raise ValueError("perception admission receipt failed registry re-verification")
    return admitted
