from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceConfidence(StrEnum):
    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"


class EvidenceItem(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=64)
    statement: str = Field(min_length=1, max_length=4000)
    confidence: EvidenceConfidence
    rule_ids: list[str] = Field(default_factory=list, max_length=64)
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SignedVerdict(StrictModel):
    grade: str = Field(pattern=r"^(A|B|C|D|F|-)$")
    signature: str = Field(min_length=8, max_length=256)
    triggered_rules: list[str] = Field(default_factory=list, max_length=128)
    summary: str = Field(min_length=1, max_length=4000)


class SecurityCase(StrictModel):
    schema_version: str = Field(default="sentinel.case.v1")
    case_id: str = Field(min_length=1, max_length=128)
    target_ref: str = Field(min_length=1, max_length=256)
    network: str = Field(min_length=1, max_length=64)
    signed_verdict: SignedVerdict
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=512)
    limitations: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> SecurityCase:
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique")
        return self


class EvidenceClaim(StrictModel):
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=64)
    confidence: EvidenceConfidence


class SentinelOpinion(StrictModel):
    schema_version: str = Field(default="sentinel.opinion.v1")
    case_id: str
    verdict_signature: str
    authority: str = Field(
        default="The signed deterministic verdict is final; this output is commentary only."
    )
    assessment: str = Field(default="EXPLANATION_ONLY")
    claims: list[EvidenceClaim] = Field(default_factory=list, max_length=128)
    limitations: list[str] = Field(default_factory=list, max_length=128)
    recommended_actions: list[str] = Field(default_factory=list, max_length=64)
    engine: str = Field(default="sentinel-baseline-v0.1")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CheckedOpinion(StrictModel):
    opinion: SentinelOpinion
    policy_violations: list[str] = Field(default_factory=list)
    accepted: bool
