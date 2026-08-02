from __future__ import annotations

from collections import defaultdict

from .models import EvidenceClaim, EvidenceConfidence, SecurityCase, SentinelOpinion


class PolicyViolation(ValueError):
    """Raised when an opinion violates the immutable evidence contract."""


def validate_opinion(case: SecurityCase, opinion: SentinelOpinion) -> list[str]:
    violations: list[str] = []
    known_ids = {item.evidence_id for item in case.evidence}

    if opinion.case_id != case.case_id:
        violations.append("case_id does not match the input case")
    if opinion.verdict_signature != case.signed_verdict.signature:
        violations.append("verdict_signature does not match the signed verdict")
    if opinion.assessment != "EXPLANATION_ONLY":
        violations.append("assessment must remain EXPLANATION_ONLY")

    for index, claim in enumerate(opinion.claims):
        unknown = sorted(set(claim.evidence_ids) - known_ids)
        if unknown:
            violations.append(f"claim[{index}] cites unknown evidence: {', '.join(unknown)}")
        if not claim.evidence_ids:
            violations.append(f"claim[{index}] has no evidence citation")

    return violations


def baseline_opinion(case: SecurityCase) -> SentinelOpinion:
    """Create a deterministic, non-generative opinion for contract testing."""

    by_rule: dict[str, list[str]] = defaultdict(list)
    confidence_by_id = {item.evidence_id: item.confidence for item in case.evidence}
    statement_by_id = {item.evidence_id: item.statement for item in case.evidence}

    for item in case.evidence:
        for rule_id in item.rule_ids:
            by_rule[rule_id].append(item.evidence_id)

    claims: list[EvidenceClaim] = []
    limitations = list(case.limitations)

    for rule_id in case.signed_verdict.triggered_rules:
        evidence_ids = by_rule.get(rule_id, [])
        if not evidence_ids:
            limitations.append(f"Triggered rule {rule_id} has no attached evidence item.")
            continue
        weakest = _weakest_confidence(confidence_by_id[evidence_id] for evidence_id in evidence_ids)
        summaries = "; ".join(statement_by_id[evidence_id] for evidence_id in evidence_ids[:3])
        claims.append(
            EvidenceClaim(
                text=f"Rule {rule_id} is supported by the supplied evidence: {summaries}",
                evidence_ids=evidence_ids,
                confidence=weakest,
            )
        )

    if not case.signed_verdict.triggered_rules:
        limitations.append("No triggered rule was supplied; Sentinel cannot create a risk claim.")

    actions = [
        "Review the cited evidence before taking action.",
        "Re-run deterministic scanning when material on-chain state changes.",
    ]

    return SentinelOpinion(
        case_id=case.case_id,
        verdict_signature=case.signed_verdict.signature,
        claims=claims,
        limitations=_deduplicate(limitations),
        recommended_actions=actions,
    )


def _weakest_confidence(values: object) -> EvidenceConfidence:
    order = {
        EvidenceConfidence.VERIFIED: 2,
        EvidenceConfidence.INFERRED: 1,
        EvidenceConfidence.UNVERIFIED: 0,
    }
    return min(values, key=order.__getitem__)  # type: ignore[arg-type]


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))
