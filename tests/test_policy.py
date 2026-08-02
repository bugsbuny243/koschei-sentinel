from koschei_sentinel.models import EvidenceClaim, EvidenceConfidence, SecurityCase
from koschei_sentinel.policy import baseline_opinion, validate_opinion


def sample_case() -> SecurityCase:
    return SecurityCase.model_validate(
        {
            "case_id": "case-1",
            "target_ref": "wallet_123",
            "network": "solana-mainnet",
            "signed_verdict": {
                "grade": "D",
                "signature": "signed-verdict-123",
                "triggered_rules": ["KS-1"],
                "summary": "Rule triggered.",
            },
            "evidence": [
                {
                    "evidence_id": "E-1",
                    "kind": "rule",
                    "statement": "Observed condition.",
                    "confidence": "VERIFIED",
                    "rule_ids": ["KS-1"],
                }
            ],
        }
    )


def test_baseline_is_policy_clean() -> None:
    case = sample_case()
    opinion = baseline_opinion(case)
    assert validate_opinion(case, opinion) == []
    assert opinion.claims[0].evidence_ids == ["E-1"]


def test_unknown_evidence_is_rejected() -> None:
    case = sample_case()
    opinion = baseline_opinion(case)
    opinion.claims.append(
        EvidenceClaim(
            text="Unsupported claim",
            evidence_ids=["E-404"],
            confidence=EvidenceConfidence.UNVERIFIED,
        )
    )
    assert "unknown evidence" in " ".join(validate_opinion(case, opinion))
