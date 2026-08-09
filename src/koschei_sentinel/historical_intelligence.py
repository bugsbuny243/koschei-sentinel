from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.models import EvidenceConfidence, StrictModel


class RelationBasis(StrEnum):
    DIRECT_ONCHAIN = "DIRECT_ONCHAIN"
    NORMALIZED_PROVIDER = "NORMALIZED_PROVIDER"
    DETERMINISTIC_DERIVATION = "DETERMINISTIC_DERIVATION"
    WATCH_ONLY_INFERENCE = "WATCH_ONLY_INFERENCE"


class ProviderStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"


class ActorRelation(StrictModel):
    schema_version: Literal["sentinel.actor-relation.v1"] = "sentinel.actor-relation.v1"
    relation_ref: str = Field(min_length=1, max_length=128)
    source_actor_ref: str = Field(min_length=1, max_length=128)
    target_actor_ref: str = Field(min_length=1, max_length=128)
    relation_kind: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    basis: RelationBasis
    confidence: EvidenceConfidence
    evidence_ids: list[str] = Field(min_length=1, max_length=128)
    first_seen_at: datetime
    last_seen_at: datetime

    @model_validator(mode="after")
    def relation_is_consistent(self) -> ActorRelation:
        if self.source_actor_ref == self.target_actor_ref:
            raise ValueError("actor relation endpoints must be different")
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot precede first_seen_at")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("actor relation evidence_ids must be unique")
        if (
            self.basis is RelationBasis.WATCH_ONLY_INFERENCE
            and self.confidence is EvidenceConfidence.VERIFIED
        ):
            raise ValueError("watch-only inference cannot be marked VERIFIED")
        return self


class FundingClusterEvent(StrictModel):
    schema_version: Literal["sentinel.funding-cluster-event.v1"] = (
        "sentinel.funding-cluster-event.v1"
    )
    event_ref: str = Field(min_length=1, max_length=128)
    cluster_ref: str = Field(min_length=1, max_length=128)
    event_kind: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    occurred_at: datetime
    actor_refs: list[str] = Field(min_length=1, max_length=256)
    target_refs: list[str] = Field(default_factory=list, max_length=256)
    evidence_ids: list[str] = Field(min_length=1, max_length=128)
    confidence: EvidenceConfidence

    @model_validator(mode="after")
    def event_refs_are_unique(self) -> FundingClusterEvent:
        _require_unique(self.actor_refs, "funding event actor_refs")
        _require_unique(self.target_refs, "funding event target_refs")
        _require_unique(self.evidence_ids, "funding event evidence_ids")
        return self


class RepeatOperatorFamily(StrictModel):
    schema_version: Literal["sentinel.repeat-operator-family.v1"] = (
        "sentinel.repeat-operator-family.v1"
    )
    family_ref: str = Field(min_length=1, max_length=128)
    operator_refs: list[str] = Field(min_length=1, max_length=256)
    case_refs: list[str] = Field(min_length=2, max_length=512)
    evidence_ids: list[str] = Field(min_length=1, max_length=256)
    first_seen_at: datetime
    last_seen_at: datetime

    @model_validator(mode="after")
    def family_is_consistent(self) -> RepeatOperatorFamily:
        _require_unique(self.operator_refs, "operator_refs")
        _require_unique(self.case_refs, "case_refs")
        _require_unique(self.evidence_ids, "repeat operator evidence_ids")
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot precede first_seen_at")
        return self


class VerdictRevision(StrictModel):
    schema_version: Literal["sentinel.verdict-revision.v1"] = "sentinel.verdict-revision.v1"
    case_ref: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1, le=1_000_000)
    verdict_signature: str = Field(min_length=8, max_length=256)
    grade: str = Field(pattern=r"^(A|B|C|D|F|-)$")
    current: bool
    supersedes_signature: str | None = Field(default=None, min_length=8, max_length=256)
    evidence_bundle_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    occurred_at: datetime

    @model_validator(mode="after")
    def revision_shape_is_valid(self) -> VerdictRevision:
        if self.revision == 1 and self.supersedes_signature is not None:
            raise ValueError("revision 1 cannot supersede an earlier signature")
        if self.revision > 1 and self.supersedes_signature is None:
            raise ValueError("revision > 1 must name the superseded signature")
        if self.supersedes_signature == self.verdict_signature:
            raise ValueError("a verdict revision cannot supersede itself")
        return self


class ProviderObservation(StrictModel):
    schema_version: Literal["sentinel.provider-observation.v1"] = (
        "sentinel.provider-observation.v1"
    )
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    query_class: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    status: ProviderStatus
    observed_at: datetime
    evidence_ids: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def provider_evidence_ids_are_unique(self) -> ProviderObservation:
        _require_unique(self.evidence_ids, "provider observation evidence_ids")
        return self


class HistoricalIntelligenceBundle(StrictModel):
    schema_version: Literal["sentinel.historical-intelligence.v1"] = (
        "sentinel.historical-intelligence.v1"
    )
    snapshot_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    actor_relations: list[ActorRelation] = Field(default_factory=list, max_length=4096)
    funding_events: list[FundingClusterEvent] = Field(default_factory=list, max_length=4096)
    incident_families: list[RepeatOperatorFamily] = Field(default_factory=list, max_length=2048)
    verdict_revisions: list[VerdictRevision] = Field(default_factory=list, max_length=8192)
    provider_observations: list[ProviderObservation] = Field(default_factory=list, max_length=4096)

    @model_validator(mode="after")
    def bundle_is_consistent(self) -> HistoricalIntelligenceBundle:
        _require_unique(
            [item.relation_ref for item in self.actor_relations],
            "actor relation refs",
        )
        _require_unique(
            [item.event_ref for item in self.funding_events],
            "funding event refs",
        )
        _require_unique(
            [item.family_ref for item in self.incident_families],
            "incident family refs",
        )
        self._validate_verdict_chains()
        return self

    def _validate_verdict_chains(self) -> None:
        by_case: dict[str, list[VerdictRevision]] = {}
        for item in self.verdict_revisions:
            by_case.setdefault(item.case_ref, []).append(item)

        for case_ref, revisions in by_case.items():
            ordered = sorted(revisions, key=lambda item: item.revision)
            revision_numbers = [item.revision for item in ordered]
            if len(revision_numbers) != len(set(revision_numbers)):
                raise ValueError(f"duplicate verdict revision for case {case_ref}")
            if revision_numbers != list(range(1, revision_numbers[-1] + 1)):
                raise ValueError(f"verdict revision chain is not contiguous for case {case_ref}")
            current = [item for item in ordered if item.current]
            if len(current) > 1:
                raise ValueError(f"multiple current verdicts for case {case_ref}")
            if current and current[0].revision != ordered[-1].revision:
                raise ValueError(f"current verdict is not latest for case {case_ref}")
            for previous, item in zip(ordered[:-1], ordered[1:], strict=True):
                if item.supersedes_signature != previous.verdict_signature:
                    raise ValueError(
                        f"verdict supersession chain is broken for case {case_ref}"
                    )


def _require_unique(values: list[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")
