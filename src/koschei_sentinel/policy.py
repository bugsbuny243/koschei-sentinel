from __future__ import annotations

from collections import defaultdict

from .models import EvidenceClaim, EvidenceConfidence, SecurityCase, SentinelOpinion

_AUTHORITY = "The signed deterministic verdict is final; this output is commentary only."
_CONFIDENCE_ORDER = {
    EvidenceConfidence.UNVERIFIED: 0,
    EvidenceConfidence.INFERRED: 1,
    EvidenceConfidence.VERIFIED: 2,
}


class PolicyViolation(ValueError):
    """Raised when an opinion violates the immutable evidence contract."""


def validate_opinion(case: SecurityCase, opinion: SentinelOpinion) -> list[str]:
    violations: list[str] = []
    known_ids = {item.evidence_id for item in case.evidence}
    confidence_by_id = {item.evidence_id: item.confidence for item in case.evidence}
    cited_ids: set[str] = set()

    if opinion.case_id != case.case_id:
        violations.append("case_id does not match the input case")
    if opinion.verdict_signature != case.signed_verdict.signature:
        violations.append("verdict_signature does not match the signed verdict")
    if opinion.authority != _AUTHORITY:
        violations.append("authority statement does not preserve deterministic finality")
    if opinion.assessment != "EXPLANATION_ONLY":
        violations.append("assessment must remain EXPLANATION_ONLY")

    for index, claim in enumerate(opinion.claims):
        cited_ids.update(claim.evidence_ids)
        unknown = sorted(set(claim.evidence_ids) - known_ids)
        if unknown:
            violations.append(f"claim[{index}] cites unknown evidence: {', '.join(unknown)}")
        if not claim.evidence_ids:
            violations.append(f"claim[{index}] has no evidence citation")
            continue
        cited_confidences = [
            confidence_by_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in confidence_by_id
        ]
        if cited_confidences and _CONFIDENCE_ORDER[claim.confidence] > min(
            _CONFIDENCE_ORDER[value] for value in cited_confidences
        ):
            violations.append(
                f"claim[{index}] confidence exceeds its weakest cited evidence"
            )

    evidence_by_rule: dict[str, set[str]] = defaultdict(set)
    for item in case.evidence:
        for rule_id in item.rule_ids:
            evidence_by_rule[rule_id].add(item.evidence_id)
    for rule_id in case.signed_verdict.triggered_rules:
        supporting_ids = evidence_by_rule.get(rule_id, set())
        if supporting_ids and not cited_ids.intersection(supporting_ids):
            violations.append(f"triggered rule {rule_id} has no cited supporting evidence")

    return violations


def baseline_opinion(case: SecurityCase) -> SentinelOpinion:
    """Create deterministic supervision without treating evidence text as instructions.

    Evidence statements are intentionally not copied into assistant text. They remain
    available in the input case and are referenced only through known evidence IDs.
    This keeps untrusted provider/on-chain text from becoming executable-looking
    supervision when it contains prompt-injection or verdict-tampering instructions.
    """

    by_rule: dict[str, list[str]] = defaultdict(list)
    confidence_by_id = {item.evidence_id: item.confidence for item in case.evidence}

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
        claims.append(
            EvidenceClaim(
                text=(
                    f"Rule {rule_id} is supported by the supplied evidence IDs: "
                    f"{', '.join(evidence_ids)}."
                ),
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
    return min(values, key=_CONFIDENCE_ORDER.__getitem__)  # type: ignore[arg-type]


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))
