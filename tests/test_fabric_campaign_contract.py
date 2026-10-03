from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from koschei_sentinel.fabric_campaign_contract import (
    CAMPAIGN_EVIDENCE_SCHEMA,
    FABRIC_OBSERVE_MODE,
    FabricCampaignEvidence,
    SentinelCampaignHypothesis,
    build_sentinel_campaign_opinion,
)


def _evidence_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CAMPAIGN_EVIDENCE_SCHEMA,
        "mode": FABRIC_OBSERVE_MODE,
        "campaign_ref": "KCAM1-FABRIC",
        "campaign_revision": 3,
        "campaign_state": "active",
        "campaign_evidence_hash_sha256": "sha256:" + "1" * 64,
        "ruleset_version": "campaign-rules.v1",
        "first_observed_at": "2026-10-02T09:00:00Z",
        "last_observed_at": "2026-10-02T10:00:00Z",
        "networks": ["ethereum", "solana"],
        "subjects": ["subject:a", "subject:b"],
        "evidence_refs": ["obs:1", "rel:1"],
        "verdict_refs": ["verdict:1"],
        "attack_path_refs": ["attack:1"],
        "missing_evidence": ["destination_owner"],
        "temporal_fingerprint_sha256": "sha256:temporal",
        "radar_fingerprint_sha256": "sha256:radar",
        "threat_fingerprint_sha256": "sha256:threat",
        "incident_refs": [
            {
                "incident_ref": "KSI-A",
                "campaign_revision": 2,
                "evidence_hash_sha256": "sha256:" + "2" * 64,
            }
        ],
        "contract_hash_sha256": "",
        "verdict_authority": False,
        "grade_authority": False,
        "containment_authority": False,
        "response_execution_authority": False,
        "same_operator_claim": False,
        "real_world_identity_claim": False,
        "wrongdoing_claim": False,
    }
    canonical = dict(payload)
    canonical["contract_hash_sha256"] = ""
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
    payload["contract_hash_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return payload


def test_campaign_evidence_accepts_valid_observe_only_contract() -> None:
    evidence = FabricCampaignEvidence.model_validate(_evidence_payload())
    assert evidence.mode == "observe"
    assert evidence.verdict_authority is False
    assert evidence.containment_authority is False
    assert evidence.response_execution_authority is False


def test_campaign_evidence_rejects_tamper_and_authority() -> None:
    tampered = _evidence_payload()
    tampered["evidence_refs"] = ["obs:1", "rel:1", "forged:1"]
    with pytest.raises(ValidationError, match="contract hash mismatch"):
        FabricCampaignEvidence.model_validate(tampered)

    authority = _evidence_payload()
    authority["verdict_authority"] = True
    with pytest.raises(ValidationError):
        FabricCampaignEvidence.model_validate(authority)


def test_sentinel_campaign_opinion_is_deterministic_and_non_authoritative() -> None:
    evidence = FabricCampaignEvidence.model_validate(_evidence_payload())
    hypotheses = [
        SentinelCampaignHypothesis(
            hypothesis_id="H2",
            summary="Bridge destination may continue interacting with related contracts.",
            evidence_refs=["rel:1"],
            confidence=0.4,
            limitations=["Observation only"],
        ),
        SentinelCampaignHypothesis(
            hypothesis_id="H1",
            summary="Observed sequence is consistent with the retained technical pathway.",
            evidence_refs=["obs:1"],
            confidence=0.6,
            limitations=["No identity claim"],
        ),
    ]
    first = build_sentinel_campaign_opinion(
        evidence,
        hypotheses=hypotheses,
        alternative_explanations=["Benign operational transfer", "Independent actors"],
        likely_next_observables=["New destination-chain interaction"],
        contradictions=["No verified common-control evidence"],
        defensive_recommendations=["Require fresh evidence before protected execution"],
        evidence_refs=["rel:1", "obs:1"],
        confidence_ceiling=0.6,
    )
    second = build_sentinel_campaign_opinion(
        evidence,
        hypotheses=list(reversed(hypotheses)),
        alternative_explanations=["Independent actors", "Benign operational transfer"],
        likely_next_observables=["New destination-chain interaction"],
        contradictions=["No verified common-control evidence"],
        defensive_recommendations=["Require fresh evidence before protected execution"],
        evidence_refs=["obs:1", "rel:1"],
        confidence_ceiling=0.6,
    )
    assert first == second
    assert first.opinion_hash_sha256.startswith("sha256:")
    assert first.verdict_authority is False
    assert first.grade_authority is False
    assert first.containment_authority is False
    assert first.response_execution_authority is False
    assert first.same_operator_claim is False
    assert first.real_world_identity_claim is False
    assert first.wrongdoing_claim is False
    assert first.missing_evidence == ["destination_owner"]


def test_sentinel_campaign_opinion_rejects_out_of_envelope_evidence_ref() -> None:
    evidence = FabricCampaignEvidence.model_validate(_evidence_payload())
    with pytest.raises(ValueError, match="outside Fabric envelope"):
        build_sentinel_campaign_opinion(evidence, evidence_refs=["not-in-envelope"])
