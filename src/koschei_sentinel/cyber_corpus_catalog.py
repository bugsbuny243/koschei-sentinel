from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import StrictModel


_DIGEST = r"^[a-f0-9]{64}$"


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


class LicenseScope(StrEnum):
    UNIFORM = "UNIFORM"
    PER_ARTIFACT = "PER_ARTIFACT"
    UNKNOWN = "UNKNOWN"


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
    license_scope: LicenseScope = LicenseScope.UNKNOWN
    license_reference: str | None = None
    acquisition_mode: str = Field(min_length=3, max_length=64)
    canonical_locator: str | None = None
    pinned_revision: str | None = None
    training_authorization: bool = False
    eval_exclusion: bool = True
    benchmark_overlap_risk: Literal[
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
        "UNKNOWN",
    ] = "UNKNOWN"
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
        if self.training_authorization and self.license_scope is LicenseScope.UNKNOWN:
            raise ValueError("training authorization requires a resolved license_scope")
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
        if (
            self.license_status in {LicenseStatus.BLOCKED, LicenseStatus.EVAL_ONLY}
            and self.training_authorization
        ):
            raise ValueError("blocked/eval-only material cannot be training-authorized")
        if self.benchmark_overlap_risk == "HIGH" and self.training_authorization:
            raise ValueError("high benchmark-overlap risk cannot be training-authorized")
        return self


class CyberArtifact(StrictModel):
    schema_version: Literal["sentinel.cyber-artifact.v3"] = "sentinel.cyber-artifact.v3"
    artifact_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._:-]{2,255}$")
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    locator: str = Field(min_length=3, max_length=2048)
    source_revision: str = Field(min_length=1, max_length=256)
    source_snapshot_sha256: str = Field(pattern=_DIGEST)
    content_sha256: str = Field(pattern=_DIGEST)
    license_status: LicenseStatus
    license_reference: str | None = None
    inherited_source_license: bool = False
    training_authorization: bool = False
    eval_exclusion: bool = True
    benchmark_overlap_risk: Literal[
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
        "UNKNOWN",
    ] = "UNKNOWN"

    @model_validator(mode="after")
    def artifact_is_fail_closed(self) -> CyberArtifact:
        allowed = {
            LicenseStatus.ALLOW_TRAINING,
            LicenseStatus.ALLOW_WITH_ATTRIBUTION,
        }
        if self.training_authorization and self.license_status not in allowed:
            raise ValueError("artifact training authorization requires approved license status")
        if self.training_authorization and not self.license_reference:
            raise ValueError("trainable artifacts require license_reference")
        if self.training_authorization and not self.eval_exclusion:
            raise ValueError("trainable artifacts must be excluded from eval material")
        if self.training_authorization and self.benchmark_overlap_risk == "HIGH":
            raise ValueError("high benchmark-overlap artifact cannot be training-authorized")
        return self


class CyberCatalogAudit(StrictModel):
    schema_version: Literal["sentinel.cyber-catalog-audit.v3"] = (
        "sentinel.cyber-catalog-audit.v3"
    )
    ready_for_collection: bool
    catalog_digest: str
    sources: int
    approved_sources: int
    training_authorized_sources: int
    distinct_domains: int
    provenance_counts: dict[str, int]
    license_counts: dict[str, int]
    license_scope_counts: dict[str, int]
    review_counts: dict[str, int]
    violations: list[str]


class CyberArtifactAudit(StrictModel):
    schema_version: Literal["sentinel.cyber-artifact-audit.v3"] = (
        "sentinel.cyber-artifact-audit.v3"
    )
    ready_for_ingestion: bool
    artifacts: int
    training_authorized_artifacts: int
    duplicate_content_sha256: list[str]
    unknown_source_ids: list[str]
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


def load_artifacts(path: str | Path) -> list[CyberArtifact]:
    rows: list[CyberArtifact] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank artifact row at line {line_number}")
        try:
            rows.append(CyberArtifact.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid artifact row at line {line_number}") from exc
    if not rows:
        raise ValueError("artifact manifest contains no entries")
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
            violations.append(
                f"training source {item.source_id} is not marked eval_exclusion"
            )

    domains = {domain for item in ordered for domain in item.domain_families}
    canonical = "\n".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in ordered
    )
    return CyberCatalogAudit(
        ready_for_collection=not violations and bool(trainable),
        catalog_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        sources=len(ordered),
        approved_sources=sum(
            item.review_status is ReviewStatus.APPROVED for item in ordered
        ),
        training_authorized_sources=len(trainable),
        distinct_domains=len(domains),
        provenance_counts=dict(
            sorted(Counter(item.provenance_tier.value for item in ordered).items())
        ),
        license_counts=dict(
            sorted(Counter(item.license_status.value for item in ordered).items())
        ),
        license_scope_counts=dict(
            sorted(Counter(item.license_scope.value for item in ordered).items())
        ),
        review_counts=dict(
            sorted(Counter(item.review_status.value for item in ordered).items())
        ),
        violations=violations,
    )


def audit_artifacts(
    artifacts: list[CyberArtifact],
    sources: list[CyberSource],
) -> CyberArtifactAudit:
    source_by_id = {source.source_id: source for source in sources}
    violations: list[str] = []
    unknown_sources = sorted({
        artifact.source_id
        for artifact in artifacts
        if artifact.source_id not in source_by_id
    })
    if unknown_sources:
        violations.append("artifacts reference unknown source_id values")

    digest_counts = Counter(artifact.content_sha256 for artifact in artifacts)
    duplicates = sorted(digest for digest, count in digest_counts.items() if count > 1)
    if duplicates:
        violations.append("duplicate artifact content detected")

    for artifact in artifacts:
        source = source_by_id.get(artifact.source_id)
        if source is None:
            continue
        if not artifact.training_authorization:
            continue
        if not source.training_authorization:
            violations.append(
                f"artifact {artifact.artifact_id} is trainable but source is not authorized"
            )
            continue
        if artifact.source_revision != source.pinned_revision:
            violations.append(
                f"artifact {artifact.artifact_id} revision does not match source pin"
            )
        if source.license_scope is LicenseScope.UNIFORM:
            if not artifact.inherited_source_license:
                violations.append(
                    f"artifact {artifact.artifact_id} must inherit uniform source license"
                )
            if artifact.license_status is not source.license_status:
                violations.append(
                    f"artifact {artifact.artifact_id} license differs from uniform source"
                )
            if artifact.license_reference != source.license_reference:
                violations.append(
                    f"artifact {artifact.artifact_id} license reference differs from source"
                )
        elif source.license_scope is LicenseScope.PER_ARTIFACT:
            if artifact.inherited_source_license:
                violations.append(
                    f"artifact {artifact.artifact_id} cannot inherit per-artifact source license"
                )
            if not artifact.license_reference:
                violations.append(
                    f"artifact {artifact.artifact_id} requires explicit artifact license"
                )
        else:
            violations.append(
                f"artifact {artifact.artifact_id} source license scope is unresolved"
            )

    return CyberArtifactAudit(
        ready_for_ingestion=not violations and any(
            artifact.training_authorization for artifact in artifacts
        ),
        artifacts=len(artifacts),
        training_authorized_artifacts=sum(
            artifact.training_authorization for artifact in artifacts
        ),
        duplicate_content_sha256=duplicates,
        unknown_source_ids=unknown_sources,
        violations=violations,
    )
