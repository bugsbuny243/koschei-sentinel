from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


class ProvenanceTier(StrEnum):
    T0_PRIMARY = "T0_PRIMARY"
    T1_AUTHORITATIVE = "T1_AUTHORITATIVE"
    T2_REVIEWED = "T2_REVIEWED"
    T3_CONTEXT_ONLY = "T3_CONTEXT_ONLY"


class LicenseStatus(StrEnum):
    ALLOW_TRAINING = "ALLOW_TRAINING"
    ALLOW_WITH_ATTRIBUTION = "ALLOW_WITH_ATTRIBUTION"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    EVAL_ONLY = "EVAL_ONLY"
    BLOCKED = "BLOCKED"


class ReviewStatus(StrEnum):
    PROPOSED = "PROPOSED"
    LICENSE_REVIEW = "LICENSE_REVIEW"
    SECURITY_REVIEW = "SECURITY_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVOKED = "REVOKED"


class CyberSource(StrictModel):
    schema_version: Literal["sentinel.cyber-source.v3"] = "sentinel.cyber-source.v3"
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    source_class: str = Field(min_length=3, max_length=128)
    domain_families: list[str] = Field(min_length=1, max_length=64)
    provenance_tier: ProvenanceTier
    license_status: LicenseStatus = LicenseStatus.REVIEW_REQUIRED
    license_reference: str | None = None
    acquisition_mode: str = Field(min_length=3, max_length=64)
    canonical_locator: str | None = None
    pinned_revision: str | None = None
    training_authorization: bool = False
    eval_exclusion: bool = True
    benchmark_overlap_risk: Literal["NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "UNKNOWN"
    review_status: ReviewStatus = ReviewStatus.PROPOSED
    notes: str | None = None

    @model_validator(mode="after")
    def approval_is_fail_closed(self) -> CyberSource:
        if len(self.domain_families) != len(set(self.domain_families)):
            raise ValueError("domain_families must be unique")
        allowed = {
            LicenseStatus.ALLOW_TRAINING,
            LicenseStatus.ALLOW_WITH_ATTRIBUTION,
        }
        if self.training_authorization and self.license_status not in allowed:
            raise ValueError("training authorization requires an approved license status")
        if self.training_authorization and self.review_status is not ReviewStatus.APPROVED:
            raise ValueError("training authorization requires APPROVED review status")
        if self.review_status is ReviewStatus.APPROVED and not self.license_reference:
            raise ValueError("approved sources require license_reference")
        if self.training_authorization and not self.canonical_locator:
            raise ValueError("trainable sources require canonical_locator")
        if self.training_authorization and not self.pinned_revision:
            raise ValueError("trainable sources require pinned_revision")
        if self.provenance_tier is ProvenanceTier.T3_CONTEXT_ONLY and self.training_authorization:
            raise ValueError("T3_CONTEXT_ONLY sources cannot be training-authorized")
        if self.license_status in {LicenseStatus.BLOCKED, LicenseStatus.EVAL_ONLY} and self.training_authorization:
            raise ValueError("blocked/eval-only material cannot be training-authorized")
        if self.benchmark_overlap_risk == "HIGH" and self.training_authorization:
            raise ValueError("high benchmark-overlap risk cannot be training-authorized")
        return self


class CyberCatalogAudit(StrictModel):
    schema_version: Literal["sentinel.cyber-catalog-audit.v3"] = "sentinel.cyber-catalog-audit.v3"
    ready_for_collection: bool
    catalog_digest: str
    sources: int
    approved_sources: int
    training_authorized_sources: int
    distinct_domains: int
    provenance_counts: dict[str, int]
    license_counts: dict[str, int]
    review_counts: dict[str, int]
    violations: list[str]


def load_catalog(path: str | Path) -> list[CyberSource]:
    rows: list[CyberSource] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank source catalog row at line {line_number}")
        try:
            rows.append(CyberSource.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid source catalog row at line {line_number}") from exc
    if not rows:
        raise ValueError("source catalog contains no entries")
    return rows


def audit_catalog(sources: list[CyberSource]) -> CyberCatalogAudit:
    ordered = sorted(sources, key=lambda item: item.source_id)
    violations: list[str] = []
    source_ids = [item.source_id for item in ordered]
    if len(source_ids) != len(set(source_ids)):
        violations.append("duplicate source_id detected")

    trainable = [item for item in ordered if item.training_authorization]
    for item in trainable:
        if item.eval_exclusion is False:
            violations.append(f"training source {item.source_id} is not marked eval_exclusion")

    domains = {domain for item in ordered for domain in item.domain_families}
    canonical = "\n".join(
        json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for item in ordered
    )
    return CyberCatalogAudit(
        ready_for_collection=not violations and bool(trainable),
        catalog_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        sources=len(ordered),
        approved_sources=sum(item.review_status is ReviewStatus.APPROVED for item in ordered),
        training_authorized_sources=len(trainable),
        distinct_domains=len(domains),
        provenance_counts=dict(sorted(Counter(item.provenance_tier.value for item in ordered).items())),
        license_counts=dict(sorted(Counter(item.license_status.value for item in ordered).items())),
        review_counts=dict(sorted(Counter(item.review_status.value for item in ordered).items())),
        violations=violations,
    )
