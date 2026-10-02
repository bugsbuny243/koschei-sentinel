from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CAMPAIGN_EVIDENCE_SCHEMA = "koschei.fabric.campaign-evidence.v1"
SENTINEL_CAMPAIGN_OPINION_SCHEMA = "koschei.fabric.sentinel-campaign-opinion.v1"
FABRIC_OBSERVE_MODE = "observe"


class FabricCampaignIncidentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_ref: str
    campaign_revision: int = Field(gt=0)
    evidence_hash_sha256: str


class FabricCampaignEvidence(BaseModel):
    """Read-only evidence envelope exported by Koschei Web3 through Fabric."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CAMPAIGN_EVIDENCE_SCHEMA]
    mode: Literal[FABRIC_OBSERVE_MODE]
    campaign_ref: str
    campaign_revision: int = Field(gt=0)
    campaign_state: str
    campaign_evidence_hash_sha256: str
    ruleset_version: str
    first_observed_at: str
    last_observed_at: str
    networks: list[str]
    subjects: list[str]
    evidence_refs: list[str]
    verdict_refs: list[str]
    attack_path_refs: list[str]
    missing_evidence: list[str]
    temporal_fingerprint_sha256: str = ""
    radar_fingerprint_sha256: str = ""
    threat_fingerprint_sha256: str = ""
    incident_refs: list[FabricCampaignIncidentRef]
    contract_hash_sha256: str
    verdict_authority: Literal[False] = False
    grade_authority: Literal[False] = False
    containment_authority: Literal[False] = False
    response_execution_authority: Literal[False] = False
    same_operator_claim: Literal[False] = False
    real_world_identity_claim: Literal[False] = False
    wrongdoing_claim: Literal[False] = False

    @model_validator(mode="after")
    def validate_boundary_and_hash(self) -> "FabricCampaignEvidence":
        if not self.campaign_ref.strip():
            raise ValueError("campaign_ref is required")
        if not _is_sha256(self.campaign_evidence_hash_sha256):
            raise ValueError("campaign evidence hash must be sha256")
        if not _is_sha256(self.contract_hash_sha256):
            raise ValueError("contract hash must be sha256")
        expected = fabric_campaign_evidence_hash(self)
        if expected != self.contract_hash_sha256:
            raise ValueError("campaign evidence contract hash mismatch")
        return self


class SentinelCampaignHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class SentinelCampaignOpinion(BaseModel):
    """Non-authoritative Sentinel interpretation of one Fabric evidence envelope."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SENTINEL_CAMPAIGN_OPINION_SCHEMA]
    mode: Literal[FABRIC_OBSERVE_MODE]
    campaign_ref: str
    campaign_revision: int = Field(gt=0)
    evidence_contract_hash_sha256: str
    hypotheses: list[SentinelCampaignHypothesis] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    likely_next_observables: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    defensive_recommendations: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence_ceiling: float = Field(ge=0.0, le=1.0)
    opinion_hash_sha256: str
    verdict_authority: Literal[False] = False
    grade_authority: Literal[False] = False
    containment_authority: Literal[False] = False
    response_execution_authority: Literal[False] = False
    same_operator_claim: Literal[False] = False
    real_world_identity_claim: Literal[False] = False
    wrongdoing_claim: Literal[False] = False

    @model_validator(mode="after")
    def validate_opinion_hash(self) -> "SentinelCampaignOpinion":
        if not _is_sha256(self.evidence_contract_hash_sha256):
            raise ValueError("evidence contract hash must be sha256")
        if sentinel_campaign_opinion_hash(self) != self.opinion_hash_sha256:
            raise ValueError("sentinel campaign opinion hash mismatch")
        return self


def build_sentinel_campaign_opinion(
    evidence: FabricCampaignEvidence,
    *,
    hypotheses: list[SentinelCampaignHypothesis] | None = None,
    alternative_explanations: list[str] | None = None,
    likely_next_observables: list[str] | None = None,
    contradictions: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    defensive_recommendations: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    confidence_ceiling: float = 0.0,
) -> SentinelCampaignOpinion:
    """Build a deterministic observe-only opinion; never a verdict or response command."""

    merged_missing = _normalized_strings([*evidence.missing_evidence, *(missing_evidence or [])])
    refs = _normalized_strings(evidence_refs or [])
    permitted_refs = set(evidence.evidence_refs) | set(evidence.verdict_refs) | set(evidence.attack_path_refs)
    if any(ref not in permitted_refs for ref in refs):
        raise ValueError("opinion references evidence outside Fabric envelope")

    hypothesis_items = sorted(
        hypotheses or [],
        key=lambda item: (item.hypothesis_id, item.summary, item.confidence),
    )
    draft = SentinelCampaignOpinion.model_construct(
        schema_version=SENTINEL_CAMPAIGN_OPINION_SCHEMA,
        mode=FABRIC_OBSERVE_MODE,
        campaign_ref=evidence.campaign_ref,
        campaign_revision=evidence.campaign_revision,
        evidence_contract_hash_sha256=evidence.contract_hash_sha256,
        hypotheses=hypothesis_items,
        alternative_explanations=_normalized_strings(alternative_explanations or []),
        likely_next_observables=_normalized_strings(likely_next_observables or []),
        contradictions=_normalized_strings(contradictions or []),
        missing_evidence=merged_missing,
        defensive_recommendations=_normalized_strings(defensive_recommendations or []),
        evidence_refs=refs,
        confidence_ceiling=max(0.0, min(float(confidence_ceiling), 1.0)),
        opinion_hash_sha256="",
        verdict_authority=False,
        grade_authority=False,
        containment_authority=False,
        response_execution_authority=False,
        same_operator_claim=False,
        real_world_identity_claim=False,
        wrongdoing_claim=False,
    )
    payload = draft.model_dump(mode="json")
    payload["opinion_hash_sha256"] = _sha256_payload(payload, clear_key="opinion_hash_sha256")
    return SentinelCampaignOpinion.model_validate(payload)


def fabric_campaign_evidence_hash(evidence: FabricCampaignEvidence) -> str:
    return _sha256_payload(evidence.model_dump(mode="json"), clear_key="contract_hash_sha256")


def sentinel_campaign_opinion_hash(opinion: SentinelCampaignOpinion) -> str:
    return _sha256_payload(opinion.model_dump(mode="json"), clear_key="opinion_hash_sha256")


def _sha256_payload(payload: dict[str, object], *, clear_key: str) -> str:
    canonical = dict(payload)
    canonical[clear_key] = ""
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    text = value.strip()
    if not text.startswith("sha256:") or len(text) != 71:
        return False
    try:
        int(text[7:], 16)
    except ValueError:
        return False
    return True


def _normalized_strings(values: list[str]) -> list[str]:
    return sorted({value.strip() for value in values if value.strip()})
