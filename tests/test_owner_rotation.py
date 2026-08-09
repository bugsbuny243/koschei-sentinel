from __future__ import annotations

import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.owner_rotation import (
    OwnerRotationBlocked,
    accept_owner_rotation,
    approve_owner_rotation_proposal,
    build_owner_rotation_checkpoint,
    build_owner_rotation_proposal,
    claim_owner_rotation,
    verify_owner_rotation_checkpoint,
)
from koschei_sentinel.shadow_baseline import (
    apply_shadow_baseline_advance,
    approve_shadow_baseline_proposal,
    build_shadow_baseline_proposal,
)
from koschei_sentinel.shadow_receipt import ShadowReplayReceipt
from koschei_sentinel.shadow_regression import build_shadow_regression_report
from koschei_sentinel.shadow_review import ShadowReviewScorecard, ShadowReviewThresholds


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _receipt(candidate: str, results: str) -> ShadowReplayReceipt:
    payload = {
        "schema_version": "sentinel.shadow-replay-receipt.v1",
        "candidate_id": candidate,
        "state": "completed_shadow_replay",
        "stage": "shadow_research_candidate",
        "authority": "explanation_only",
        "plan_digest": hashlib.sha256(f"plan:{candidate}".encode()).hexdigest(),
        "proposal_digest": hashlib.sha256(f"proposal:{candidate}".encode()).hexdigest(),
        "approval_digest": hashlib.sha256(f"approval:{candidate}".encode()).hexdigest(),
        "replay_sha256": "a" * 64,
        "replay_cases": 2,
        "results_path": f"build/shadow/{candidate}/results.jsonl",
        "results_sha256": results,
        "results_cases": 2,
        "output_dir": f"build/shadow/{candidate}",
        "complete_case_coverage": True,
        "ordered_case_identity_match": True,
        "manual_review_required": True,
        "benchmark_recheck_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReplayReceipt.model_validate({**payload, "receipt_digest": _digest(payload)})


def _scorecard(receipt: ShadowReplayReceipt) -> ShadowReviewScorecard:
    payload = {
        "schema_version": "sentinel.shadow-review-scorecard.v1",
        "candidate_id": receipt.candidate_id,
        "state": "reviewed_shadow_replay",
        "authority": "explanation_only",
        "receipt_digest": receipt.receipt_digest,
        "plan_digest": receipt.plan_digest,
        "results_sha256": receipt.results_sha256,
        "review_sha256": hashlib.sha256(f"review:{receipt.candidate_id}".encode()).hexdigest(),
        "reviewers": ["owner-review@koschei"],
        "total_cases": 2,
        "passed_cases": 2,
        "failed_cases": 0,
        "case_pass_rate": 1.0,
        "authority_score": 1.0,
        "grounding_score": 1.0,
        "abstention_score": 1.0,
        "privacy_score": 1.0,
        "followup_cases": [],
        "thresholds": ShadowReviewThresholds().model_dump(mode="json"),
        "gate_passed": True,
        "complete_manual_review": True,
        "benchmark_recheck_required": True,
        "owner_decision_required": True,
        "automatic_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "production_deployment_allowed": False,
        "web3_runtime_integration_allowed": False,
        "verdict_mutation_allowed": False,
    }
    return ShadowReviewScorecard.model_validate({**payload, "scorecard_digest": _digest(payload)})


def _lineage(owner: Ed25519PrivateKey):
    baseline_receipt = _receipt("candidate-a", "1" * 64)
    candidate_receipt = _receipt("candidate-b", "2" * 64)
    baseline_score = _scorecard(baseline_receipt)
    candidate_score = _scorecard(candidate_receipt)
    report = build_shadow_regression_report(
        baseline_score,
        baseline_receipt,
        candidate_score,
        candidate_receipt,
    )
    proposal = build_shadow_baseline_proposal(
        report,
        baseline_score,
        baseline_receipt,
        candidate_score,
        candidate_receipt,
        owner.public_key(),
    )
    approval = approve_shadow_baseline_proposal(
        proposal,
        owner,
        approver_id="owner@koschei",
    )
    return apply_shadow_baseline_advance(
        proposal,
        approval,
        report,
        baseline_score,
        baseline_receipt,
        candidate_score,
        candidate_receipt,
        owner.public_key(),
    )


def _checkpoint(owner: Ed25519PrivateKey, successor: Ed25519PrivateKey):
    lineage = _lineage(owner)
    proposal = build_owner_rotation_proposal(
        lineage,
        owner.public_key(),
        successor.public_key(),
    )
    approval = approve_owner_rotation_proposal(
        proposal,
        owner,
        approver_id="owner@koschei",
    )
    acceptance = accept_owner_rotation(
        proposal,
        approval,
        owner.public_key(),
        successor,
        accepter_id="successor@koschei",
    )
    checkpoint = build_owner_rotation_checkpoint(
        proposal,
        approval,
        acceptance,
        lineage,
        owner.public_key(),
        successor.public_key(),
    )
    return lineage, checkpoint


def test_dual_signed_rotation_checkpoint_is_non_activating() -> None:
    owner = Ed25519PrivateKey.generate()
    successor = Ed25519PrivateKey.generate()
    lineage, checkpoint = _checkpoint(owner, successor)
    verified = verify_owner_rotation_checkpoint(
        checkpoint,
        lineage,
        owner.public_key(),
        successor.public_key(),
    )
    assert verified.dual_signature_verified is True
    assert verified.baseline_rekey_required is True
    assert verified.automatic_key_activation_allowed is False
    assert verified.production_deployment_allowed is False


def test_lineage_rejection_is_translated_to_rotation_block() -> None:
    owner = Ed25519PrivateKey.generate()
    successor = Ed25519PrivateKey.generate()
    lineage = _lineage(owner)
    with pytest.raises(OwnerRotationBlocked, match="baseline lineage verification failed"):
        build_owner_rotation_proposal(
            lineage,
            Ed25519PrivateKey.generate().public_key(),
            successor.public_key(),
        )


def test_wrong_next_key_and_same_key_rotation_fail_closed() -> None:
    owner = Ed25519PrivateKey.generate()
    successor = Ed25519PrivateKey.generate()
    lineage, checkpoint = _checkpoint(owner, successor)
    with pytest.raises(OwnerRotationBlocked, match="checkpoint next owner key"):
        verify_owner_rotation_checkpoint(
            checkpoint,
            lineage,
            owner.public_key(),
            Ed25519PrivateKey.generate().public_key(),
        )
    with pytest.raises(OwnerRotationBlocked, match="must differ"):
        build_owner_rotation_proposal(lineage, owner.public_key(), owner.public_key())


def test_approver_identity_tamper_breaks_current_owner_signature() -> None:
    owner = Ed25519PrivateKey.generate()
    successor = Ed25519PrivateKey.generate()
    lineage, checkpoint = _checkpoint(owner, successor)
    payload = checkpoint.model_dump(mode="json")
    payload["current_approver_id"] = "attacker@koschei"
    forged_approval_payload = {
        "schema_version": "sentinel.owner-key-rotation-current-approval.v1",
        "state": "current_owner_signed_handoff",
        "authority": "governance_evidence_only",
        "approver_id": payload["current_approver_id"],
        "current_owner_key_fingerprint": payload["current_owner_key_fingerprint"],
        "next_owner_key_fingerprint": payload["next_owner_key_fingerprint"],
        "proposal_digest": payload["proposal_digest"],
        "signature_algorithm": "ed25519",
        "signature_base64": payload["current_owner_signature_base64"],
        "signature_verified": True,
        "automatic_key_activation_allowed": False,
        "production_deployment_allowed": False,
    }
    payload["current_approval_digest"] = _digest(forged_approval_payload)
    payload.pop("checkpoint_digest")
    tampered = checkpoint.model_validate({**payload, "checkpoint_digest": _digest(payload)})
    with pytest.raises(OwnerRotationBlocked, match="current owner rotation signature"):
        verify_owner_rotation_checkpoint(
            tampered,
            lineage,
            owner.public_key(),
            successor.public_key(),
        )


def test_claim_verifies_checkpoint_before_writing(tmp_path) -> None:
    owner = Ed25519PrivateKey.generate()
    successor = Ed25519PrivateKey.generate()
    lineage, checkpoint = _checkpoint(owner, successor)
    payload = checkpoint.model_dump(mode="json")
    payload["current_approver_id"] = "attacker@koschei"
    forged_approval_payload = {
        "schema_version": "sentinel.owner-key-rotation-current-approval.v1",
        "state": "current_owner_signed_handoff",
        "authority": "governance_evidence_only",
        "approver_id": payload["current_approver_id"],
        "current_owner_key_fingerprint": payload["current_owner_key_fingerprint"],
        "next_owner_key_fingerprint": payload["next_owner_key_fingerprint"],
        "proposal_digest": payload["proposal_digest"],
        "signature_algorithm": "ed25519",
        "signature_base64": payload["current_owner_signature_base64"],
        "signature_verified": True,
        "automatic_key_activation_allowed": False,
        "production_deployment_allowed": False,
    }
    payload["current_approval_digest"] = _digest(forged_approval_payload)
    payload.pop("checkpoint_digest")
    tampered = checkpoint.model_validate({**payload, "checkpoint_digest": _digest(payload)})

    with pytest.raises(OwnerRotationBlocked, match="current owner rotation signature"):
        claim_owner_rotation(
            tampered,
            lineage,
            owner.public_key(),
            successor.public_key(),
            tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_canonical_claim_rejects_two_successors_for_same_lineage(tmp_path) -> None:
    owner = Ed25519PrivateKey.generate()
    first = Ed25519PrivateKey.generate()
    second = Ed25519PrivateKey.generate()
    lineage, first_checkpoint = _checkpoint(owner, first)
    claim_owner_rotation(
        first_checkpoint,
        lineage,
        owner.public_key(),
        first.public_key(),
        tmp_path,
    )
    claim_owner_rotation(
        first_checkpoint,
        lineage,
        owner.public_key(),
        first.public_key(),
        tmp_path,
    )

    proposal = build_owner_rotation_proposal(lineage, owner.public_key(), second.public_key())
    approval = approve_owner_rotation_proposal(proposal, owner, approver_id="owner@koschei")
    acceptance = accept_owner_rotation(
        proposal,
        approval,
        owner.public_key(),
        second,
        accepter_id="second@koschei",
    )
    second_checkpoint = build_owner_rotation_checkpoint(
        proposal,
        approval,
        acceptance,
        lineage,
        owner.public_key(),
        second.public_key(),
    )
    with pytest.raises(OwnerRotationBlocked, match="different owner rotation claim"):
        claim_owner_rotation(
            second_checkpoint,
            lineage,
            owner.public_key(),
            second.public_key(),
            tmp_path,
        )
