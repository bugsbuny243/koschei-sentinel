from __future__ import annotations

from typing import Literal

from pydantic import Field

from koschei_sentinel.cyber_perception import PerceptionBatch
from koschei_sentinel.models import StrictModel
from koschei_sentinel.perception_fusion import (
    PerceptionFusionReceipt,
    fuse_perception_batches,
)
from koschei_sentinel.perception_source_registry import (
    AdmittedPerceptionBatch,
    PerceptionSourceRegistry,
    perception_registry_sha256,
    verify_admitted_perception_batch,
)


class PerceptionAssuranceSummary(StrictModel):
    schema_version: Literal["sentinel.perception-assurance-summary.v1"] = (
        "sentinel.perception-assurance-summary.v1"
    )
    registry_id: str
    registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    fused_batch_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    admission_ids: list[str]
    adapter_ids: list[str]
    source_principals: list[str]
    independence_domains: list[str]
    independent_domain_count: int = Field(ge=1)
    observations_per_independence_domain: dict[str, int]
    largest_domain_fraction: float = Field(ge=0.0, le=1.0)
    production_admitted: Literal[True] = True


class AssuredPerceptionFusionResult(StrictModel):
    schema_version: Literal["sentinel.assured-perception-fusion-result.v1"] = (
        "sentinel.assured-perception-fusion-result.v1"
    )
    perception_batch: PerceptionBatch
    fusion_receipt: PerceptionFusionReceipt
    assurance: PerceptionAssuranceSummary


def fuse_admitted_perception_batches(
    admitted_batches: list[AdmittedPerceptionBatch],
    *,
    registry: PerceptionSourceRegistry,
    fused_batch_id: str,
) -> AssuredPerceptionFusionResult:
    if not admitted_batches:
        raise ValueError("assured perception fusion requires admitted input batches")

    verified = [
        verify_admitted_perception_batch(row, registry)
        for row in admitted_batches
    ]
    admission_ids = [row.admission.admission_id for row in verified]
    if len(admission_ids) != len(set(admission_ids)):
        raise ValueError("assured perception fusion admission_id values must be unique")
    adapter_batch_ids = [row.admission.adapter_batch_sha256 for row in verified]
    if len(adapter_batch_ids) != len(set(adapter_batch_ids)):
        raise ValueError("assured perception fusion rejects replayed adapter batches")

    fusion = fuse_perception_batches(
        [row.perception_batch for row in verified],
        fused_batch_id=fused_batch_id,
    )

    enrollment_by_principal = {row.principal: row for row in registry.enrollments}
    observations_per_domain: dict[str, int] = {}
    for observation in fusion.perception_batch.observations:
        principal = f"{observation.source_type.value}:{observation.source_instance}"
        enrollment = enrollment_by_principal.get(principal)
        if enrollment is None:
            raise ValueError(f"fused perception source is absent from registry: {principal}")
        domain = enrollment.independence_domain
        observations_per_domain[domain] = observations_per_domain.get(domain, 0) + 1

    total = len(fusion.perception_batch.observations)
    largest = max(observations_per_domain.values()) / total
    domains = sorted(observations_per_domain)
    assurance = PerceptionAssuranceSummary(
        registry_id=registry.registry_id,
        registry_sha256=perception_registry_sha256(registry),
        fused_batch_sha256=fusion.receipt.fused_batch_sha256,
        admission_ids=sorted(admission_ids),
        adapter_ids=sorted({row.admission.adapter_id for row in verified}),
        source_principals=fusion.receipt.source_principals,
        independence_domains=domains,
        independent_domain_count=len(domains),
        observations_per_independence_domain=dict(sorted(observations_per_domain.items())),
        largest_domain_fraction=largest,
    )
    return AssuredPerceptionFusionResult(
        perception_batch=fusion.perception_batch,
        fusion_receipt=fusion.receipt,
        assurance=assurance,
    )
