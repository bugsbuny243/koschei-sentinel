from __future__ import annotations

from koschei_sentinel.models import (
    EvidenceClaim,
    EvidenceConfidence,
    EvidenceItem,
    SecurityCase,
    SentinelOpinion,
    SignedVerdict,
)
from koschei_sentinel.policy import validate_opinion
from koschei_sentinel.sealing import seal_candidate_opinion


def _case() -> SecurityCase:
    return SecurityCase(
        case_id="case_sealing_fixture",
        target_ref="target_sealing_fixture",
        network="solana-mainnet",
        signed_verdict=SignedVerdict(
            grade="D",
            signature="signature_sealing_fixture",
            triggered_rules=["KS-RULE-001", "KS-RULE-002"],
            summary="Two deterministic rules were triggered.",
        ),
        evidence=[
            EvidenceItem(
                evidence_id="E-001",
                kind="holder_intelligence",
                statement="A bounded holder concentration condition was observed.",
                confidence=EvidenceConfidence.INFERRED,
                rule_ids=["KS-RULE-001"],
            ),
            EvidenceItem(
                evidence_id="E-002",
                kind="authority_state",
                statement="A deterministic authority condition was verified.",
                confidence=EvidenceConfidence.VERIFIED,
                rule_ids=["KS-RULE-002"],
            ),
        ],
        limitations=["No live market data was supplied."],
    )


def test_invalid_json_falls_back_to_grounded_baseline() -> None:
    case = _case()

    result = seal_candidate_opinion(case, "{not-json")

    assert result.accepted
    assert result.candidate_parse_error
    assert result.accepted_claims == 0
    assert result.fallback_rules == ["KS-RULE-001", "KS-RULE-002"]
    assert not result.policy_violations
    assert not validate_opinion(case, result.opinion)


def test_sealer_locks_identity_downgrades_confidence_and_covers_rules() -> None:
    case = _case()
    candidate = {
        "case_id": "wrong-case",
        "verdict_signature": "wrong-signature",
        "authority": "The model is final.",
        "assessment": "OVERRIDE",
        "claims": [
            {
                "text": "The first rule is supported by the cited evidence.",
                "evidence_ids": ["E-001", "E-404"],
                "confidence": "VERIFIED",
            },
            {
                "text": "This claim has no supplied evidence.",
                "evidence_ids": ["E-404"],
                "confidence": "VERIFIED",
            },
        ],
    }

    result = seal_candidate_opinion(case, candidate)

    assert result.accepted
    assert result.accepted_claims == 1
    assert result.dropped_claims == 1
    assert result.fallback_rules == ["KS-RULE-002"]
    assert result.opinion.case_id == case.case_id
    assert result.opinion.verdict_signature == case.signed_verdict.signature
    assert result.opinion.assessment == "EXPLANATION_ONLY"
    assert result.opinion.claims[0].evidence_ids == ["E-001"]
    assert result.opinion.claims[0].confidence == EvidenceConfidence.INFERRED
    assert "locked_case_id" in result.repairs
    assert "locked_verdict_signature" in result.repairs
    assert "claim[0]_confidence_downgraded" in result.repairs
    assert not validate_opinion(case, result.opinion)


def test_policy_rejects_confidence_escalation_and_missing_rule_coverage() -> None:
    case = _case()
    opinion = SentinelOpinion(
        case_id=case.case_id,
        verdict_signature=case.signed_verdict.signature,
        claims=[
            EvidenceClaim(
                text="The first rule is certain.",
                evidence_ids=["E-001"],
                confidence=EvidenceConfidence.VERIFIED,
            )
        ],
    )

    violations = validate_opinion(case, opinion)

    assert any("confidence exceeds" in item for item in violations)
    assert any("KS-RULE-002" in item for item in violations)


def test_no_rule_case_cannot_gain_model_risk_claims() -> None:
    case = _case().model_copy(
        update={
            "signed_verdict": SignedVerdict(
                grade="B",
                signature="signature_no_rule_fixture",
                triggered_rules=[],
                summary="No deterministic rule was triggered.",
            )
        }
    )
    candidate = {
        "claims": [
            {
                "text": "The model invented a risk claim.",
                "evidence_ids": ["E-001"],
                "confidence": "INFERRED",
            }
        ]
    }

    result = seal_candidate_opinion(case, candidate)

    assert result.accepted
    assert result.accepted_claims == 0
    assert result.dropped_claims == 1
    assert result.opinion.claims == []
    assert not validate_opinion(case, result.opinion)
