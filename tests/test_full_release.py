from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.full_release import (
    CanaryEvidence,
    FullReleaseBlocked,
    approve_full_release,
    build_full_release_proposal,
    verify_full_release,
)
from koschei_sentinel.production_authority import (
    ProductionAuthority,
    ProductionAuthorityProposal,
    approve_production_authority,
    canonical_json_digest,
)
from koschei_sentinel.promotion import public_key_fingerprint


DIGEST = "a" * 64


def _canary_authority(
    private_key: Ed25519PrivateKey,
) -> tuple[ProductionAuthorityProposal, ProductionAuthority]:
    proposal_payload = {
        "schema_version": "sentinel.production-authority-proposal.v1",
        "candidate_id": "candidate-1",
        "finalization_digest": DIGEST,
        "holdout_evidence_digests": [DIGEST],
        "deployment_scope": "canary_only",
        "max_initial_traffic_percent": 5,
        "rollback_required": True,
        "emergency_disable_required": True,
        "automatic_expansion_allowed": False,
        "owner_key_fingerprint": public_key_fingerprint(private_key.public_key()),
    }
    proposal = ProductionAuthorityProposal.model_validate(
        {
            **proposal_payload,
            "proposal_digest": canonical_json_digest(proposal_payload),
        }
    )
    authority = approve_production_authority(
        proposal,
        private_key,
        approver_id="owner",
    )
    return proposal, authority


def _evidence(authority: ProductionAuthority, **overrides) -> CanaryEvidence:
    payload = {
        "candidate_id": authority.candidate_id,
        "production_authority_digest": authority.authority_digest,
        "observed_request_count": 1000,
        "error_rate": 0.005,
        "critical_incident_count": 0,
        "rollback_drill_passed": True,
        "emergency_disable_drill_passed": True,
        "telemetry_complete": True,
    }
    payload.update(overrides)
    return CanaryEvidence(**payload)


def test_full_release_requires_healthy_canary_and_owner_signature() -> None:
    key = Ed25519PrivateKey.generate()
    canary_proposal, authority = _canary_authority(key)
    proposal = build_full_release_proposal(
        canary_proposal, authority, _evidence(authority), key.public_key()
    )
    release = approve_full_release(proposal, key, approver_id="owner")

    verified = verify_full_release(proposal, release, key.public_key())

    assert verified.production_deployment_allowed is True
    assert verified.rollout_scope == "general_release"
    assert verified.rollback_required is True
    assert verified.emergency_disable_required is True


def test_full_release_blocks_bad_canary_error_rate() -> None:
    key = Ed25519PrivateKey.generate()
    canary_proposal, authority = _canary_authority(key)

    with pytest.raises(FullReleaseBlocked, match="error rate"):
        build_full_release_proposal(
            canary_proposal,
            authority,
            _evidence(authority, error_rate=0.03),
            key.public_key(),
        )


def test_full_release_blocks_missing_rollback_drill() -> None:
    key = Ed25519PrivateKey.generate()
    canary_proposal, authority = _canary_authority(key)

    with pytest.raises(FullReleaseBlocked, match="rollback drill"):
        build_full_release_proposal(
            canary_proposal,
            authority,
            _evidence(authority, rollback_drill_passed=False),
            key.public_key(),
        )
