from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from .models import (
    EvidenceClaim,
    EvidenceConfidence,
    SecurityCase,
    SentinelOpinion,
    StrictModel,
)
from .policy import baseline_opinion, validate_opinion

_AUTHORITY = "The signed deterministic verdict is final; this output is commentary only."
_CONFIDENCE_ORDER = {
    EvidenceConfidence.UNVERIFIED: 0,
    EvidenceConfidence.INFERRED: 1,
    EvidenceConfidence.VERIFIED: 2,
}


class SealResult(StrictModel):
    schema_version: Literal["sentinel.seal-result.v1"] = "sentinel.seal-result.v1"
    opinion: SentinelOpinion
    candidate_parse_error: bool = False
    accepted_claims: int = Field(default=0, ge=0, le=128)
    dropped_claims: int = Field(default=0, ge=0, le=10000)
    fallback_rules: list[str] = Field(default_factory=list, max_length=128)
    repairs: list[str] = Field(default_factory=list, max_length=256)
    policy_violations: list[str] = Field(default_factory=list, max_length=256)
    accepted: bool


def seal_candidate_opinion(
    case: SecurityCase,
    candidate: str | Mapping[str, Any] | SentinelOpinion | None,
    *,
    engine: str = "koschei-sentinel-sealed-v0.9",
) -> SealResult:
    """Lock immutable fields and fail closed around untrusted model commentary."""

    baseline = baseline_opinion(case)
    payload, parse_error = _parse_candidate(candidate)
    repairs: list[str] = []
    if parse_error:
        repairs.append("candidate_parse_fallback")
    repairs.extend(_identity_repairs(case, payload))

    evidence_by_id = {item.evidence_id: item for item in case.evidence}
    evidence_by_rule: dict[str, set[str]] = {}
    for item in case.evidence:
        for rule_id in item.rule_ids:
            evidence_by_rule.setdefault(rule_id, set()).add(item.evidence_id)
    allowed_ids = {
        evidence_id
        for rule_id in case.signed_verdict.triggered_rules
        for evidence_id in evidence_by_rule.get(rule_id, set())
    }

    raw_claims = payload.get("claims", []) if payload is not None else []
    if not isinstance(raw_claims, list):
        raw_claims = []
        repairs.append("candidate_claims_not_a_list")

    accepted_claims: list[EvidenceClaim] = []
    dropped_claims = 0
    seen_claims: set[tuple[str, tuple[str, ...]]] = set()

    for index, raw_claim in enumerate(raw_claims):
        claim = _seal_claim(
            raw_claim,
            evidence_by_id=evidence_by_id,
            allowed_ids=allowed_ids,
            repairs=repairs,
            index=index,
        )
        if claim is None:
            dropped_claims += 1
            continue
        key = (claim.text, tuple(claim.evidence_ids))
        if key in seen_claims:
            dropped_claims += 1
            repairs.append(f"claim[{index}]_duplicate_dropped")
            continue
        seen_claims.add(key)
        accepted_claims.append(claim)

    covered_ids = {
        evidence_id
        for claim in accepted_claims
        for evidence_id in claim.evidence_ids
    }
    baseline_by_rule = _baseline_claims_by_rule(case, baseline)
    fallback_rules: list[str] = []

    for rule_id in case.signed_verdict.triggered_rules:
        supporting_ids = evidence_by_rule.get(rule_id, set())
        if not supporting_ids or covered_ids.intersection(supporting_ids):
            continue
        fallback = baseline_by_rule.get(rule_id)
        if fallback is not None:
            accepted_claims.append(fallback)
            covered_ids.update(fallback.evidence_ids)
            fallback_rules.append(rule_id)
            repairs.append(f"rule_{rule_id}_deterministic_fallback")

    opinion = SentinelOpinion(
        case_id=case.case_id,
        verdict_signature=case.signed_verdict.signature,
        authority=_AUTHORITY,
        assessment="EXPLANATION_ONLY",
        claims=accepted_claims,
        limitations=baseline.limitations,
        recommended_actions=baseline.recommended_actions,
        engine=engine,
    )
    violations = validate_opinion(case, opinion)

    if violations:
        repairs.append("final_policy_fallback")
        opinion = baseline.model_copy(update={"engine": engine})
        violations = validate_opinion(case, opinion)
        dropped_claims += len(accepted_claims)
        accepted_count = 0
        fallback_rules = list(case.signed_verdict.triggered_rules)
    else:
        accepted_count = len(accepted_claims) - len(fallback_rules)

    return SealResult(
        opinion=opinion,
        candidate_parse_error=parse_error,
        accepted_claims=accepted_count,
        dropped_claims=dropped_claims,
        fallback_rules=list(dict.fromkeys(fallback_rules)),
        repairs=list(dict.fromkeys(repairs)),
        policy_violations=violations,
        accepted=not violations,
    )


def _parse_candidate(
    candidate: str | Mapping[str, Any] | SentinelOpinion | None,
) -> tuple[dict[str, Any] | None, bool]:
    if candidate is None:
        return None, True
    if isinstance(candidate, SentinelOpinion):
        return candidate.model_dump(mode="json"), False
    if isinstance(candidate, Mapping):
        return dict(candidate), False
    if isinstance(candidate, str):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None, True
        if isinstance(parsed, Mapping):
            return dict(parsed), False
    return None, True


def _identity_repairs(
    case: SecurityCase,
    payload: Mapping[str, Any] | None,
) -> list[str]:
    if payload is None:
        return []
    expected = {
        "case_id": case.case_id,
        "verdict_signature": case.signed_verdict.signature,
        "authority": _AUTHORITY,
        "assessment": "EXPLANATION_ONLY",
    }
    return [
        f"locked_{key}"
        for key, value in expected.items()
        if key in payload and payload[key] != value
    ]


def _seal_claim(
    raw_claim: Any,
    *,
    evidence_by_id: dict[str, Any],
    allowed_ids: set[str],
    repairs: list[str],
    index: int,
) -> EvidenceClaim | None:
    if not isinstance(raw_claim, Mapping):
        repairs.append(f"claim[{index}]_not_an_object")
        return None
    text = raw_claim.get("text")
    if not isinstance(text, str) or not text.strip() or len(text.strip()) > 4000:
        repairs.append(f"claim[{index}]_invalid_text")
        return None
    raw_ids = raw_claim.get("evidence_ids")
    if not isinstance(raw_ids, list):
        repairs.append(f"claim[{index}]_invalid_evidence_ids")
        return None
    evidence_ids = list(
        dict.fromkeys(
            value
            for value in raw_ids
            if isinstance(value, str) and value in allowed_ids
        )
    )
    if not evidence_ids:
        repairs.append(f"claim[{index}]_unsupported_evidence_dropped")
        return None
    try:
        confidence = EvidenceConfidence(raw_claim.get("confidence"))
    except ValueError:
        repairs.append(f"claim[{index}]_invalid_confidence")
        return None
    weakest = min(
        (evidence_by_id[evidence_id].confidence for evidence_id in evidence_ids),
        key=_CONFIDENCE_ORDER.__getitem__,
    )
    if _CONFIDENCE_ORDER[confidence] > _CONFIDENCE_ORDER[weakest]:
        confidence = weakest
        repairs.append(f"claim[{index}]_confidence_downgraded")
    return EvidenceClaim(
        text=text.strip(),
        evidence_ids=evidence_ids,
        confidence=confidence,
    )


def _baseline_claims_by_rule(
    case: SecurityCase,
    baseline: SentinelOpinion,
) -> dict[str, EvidenceClaim]:
    output: dict[str, EvidenceClaim] = {}
    for rule_id in case.signed_verdict.triggered_rules:
        supporting_ids = {
            item.evidence_id for item in case.evidence if rule_id in item.rule_ids
        }
        for claim in baseline.claims:
            if supporting_ids.intersection(claim.evidence_ids):
                output[rule_id] = claim
                break
    return output
