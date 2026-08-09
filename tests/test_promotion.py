from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from koschei_sentinel.candidate_finalization import CandidateFinalization
from koschei_sentinel.promotion import (
    PromotionApproval,
    PromotionBlocked,
    PromotionPolicy,
    approve_promotion_proposal,
    build_promotion_proposal,
    load_promotion_approval,
    load_promotion_proposal,
    verify_promotion_approval,
    write_artifact,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _finalization() -> CandidateFinalization:
    payload = {
        "schema_version": "sentinel.candidate-finalization.v1",
        "candidate_id": "sentinel-shadow-v1",
        "state": "finalized_incubation",
        "authority": "explanation_only",
        "offline_receipt_digest": "1" * 64,
        "job_digest": "2" * 64,
        "adapter_manifest_digest": "3" * 64,
        "adapter_digest": "4" * 64,
        "dataset_manifest_digest": "5" * 64,
        "training_config_digest": "6" * 64,
        "comparison_digest": "7" * 64,
        "benchmark_suite_digest": "8" * 64,
        "benchmark_report_digest": "9" * 64,
        "candidate_record_digest": "a" * 64,
        "previous_registry_digest": "b" * 64,
        "updated_registry_digest": "c" * 64,
        "automatic_registry_replacement_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
    }
    return CandidateFinalization.model_validate(
        {**payload, "finalization_digest": _digest(payload)}
    )


def _policy(*, suite_digest: str = "8" * 64) -> PromotionPolicy:
    return PromotionPolicy(
        policy_id="shadow-research-2026-08",
        required_benchmark_suite_digest=suite_digest,
    )


def _keys(tmp_path: Path) -> tuple[Ed25519PrivateKey, Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "owner-private.pem"
    public_path = tmp_path / "owner-public.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private, private_path, public_path


def test_owner_signature_approves_shadow_research_only(tmp_path: Path) -> None:
    private, _, _ = _keys(tmp_path)
    policy = _policy()
    proposal = build_promotion_proposal(_finalization(), private.public_key(), policy)
    approval = approve_promotion_proposal(
        proposal,
        private,
        policy,
        approver_id="owner@koschei",
    )

    verified = verify_promotion_approval(
        proposal,
        approval,
        private.public_key(),
        policy,
    )

    assert verified.state == "owner_approved_shadow_research"
    assert verified.approved_stage == "shadow_research_candidate"
    assert verified.authority == "explanation_only"
    assert verified.signature_verified is True
    assert verified.automatic_promotion_allowed is False
    assert verified.automatic_deployment_allowed is False
    assert verified.production_deployment_allowed is False
    assert verified.web3_runtime_integration_allowed is False


def test_unapproved_benchmark_suite_cannot_create_proposal(tmp_path: Path) -> None:
    private, _, _ = _keys(tmp_path)
    with pytest.raises(PromotionBlocked, match="benchmark suite"):
        build_promotion_proposal(
            _finalization(),
            private.public_key(),
            _policy(suite_digest="f" * 64),
        )


def test_stale_policy_cannot_approve_existing_proposal(tmp_path: Path) -> None:
    private, _, _ = _keys(tmp_path)
    proposal = build_promotion_proposal(
        _finalization(),
        private.public_key(),
        _policy(),
    )
    changed_policy = _policy()
    changed_policy = changed_policy.model_copy(update={"policy_id": "shadow-research-next"})

    with pytest.raises(PromotionBlocked, match="does not match the supplied policy"):
        approve_promotion_proposal(
            proposal,
            private,
            changed_policy,
            approver_id="owner",
        )


def test_wrong_private_key_cannot_approve_proposal(tmp_path: Path) -> None:
    owner, _, _ = _keys(tmp_path)
    attacker = Ed25519PrivateKey.generate()
    policy = _policy()
    proposal = build_promotion_proposal(_finalization(), owner.public_key(), policy)

    with pytest.raises(PromotionBlocked, match="does not match proposal"):
        approve_promotion_proposal(
            proposal,
            attacker,
            policy,
            approver_id="not-owner",
        )


def test_proposal_tampering_is_detected_before_signing(tmp_path: Path) -> None:
    private, _, _ = _keys(tmp_path)
    policy = _policy()
    proposal = build_promotion_proposal(_finalization(), private.public_key(), policy)
    tampered = proposal.model_copy(update={"adapter_digest": "f" * 64})

    with pytest.raises(PromotionBlocked, match="digest"):
        approve_promotion_proposal(
            tampered,
            private,
            policy,
            approver_id="owner",
        )


def test_signature_cannot_be_reused_for_another_proposal(tmp_path: Path) -> None:
    private, _, _ = _keys(tmp_path)
    policy = _policy()
    proposal = build_promotion_proposal(_finalization(), private.public_key(), policy)
    approval = approve_promotion_proposal(
        proposal,
        private,
        policy,
        approver_id="owner",
    )
    changed = proposal.model_copy(
        update={
            "benchmark_report_digest": "e" * 64,
            "proposal_digest": "d" * 64,
        }
    )

    with pytest.raises(PromotionBlocked):
        verify_promotion_approval(changed, approval, private.public_key(), policy)


def test_schema_cannot_enable_production_deployment(tmp_path: Path) -> None:
    private, _, _ = _keys(tmp_path)
    policy = _policy()
    proposal = build_promotion_proposal(_finalization(), private.public_key(), policy)
    approval = approve_promotion_proposal(
        proposal,
        private,
        policy,
        approver_id="owner",
    )
    payload = approval.model_dump(mode="json")
    payload["production_deployment_allowed"] = True

    with pytest.raises(ValidationError):
        PromotionApproval.model_validate(payload)


def test_written_artifacts_are_digest_checked_on_load(tmp_path: Path) -> None:
    private, _, _ = _keys(tmp_path)
    policy = _policy()
    proposal = build_promotion_proposal(_finalization(), private.public_key(), policy)
    approval = approve_promotion_proposal(
        proposal,
        private,
        policy,
        approver_id="owner",
    )
    proposal_path = tmp_path / "proposal.json"
    approval_path = tmp_path / "approval.json"
    write_artifact(proposal, proposal_path)
    write_artifact(approval, approval_path)

    assert load_promotion_proposal(proposal_path) == proposal
    assert load_promotion_approval(approval_path) == approval

    payload = json.loads(approval_path.read_text(encoding="utf-8"))
    payload["approver_id"] = "attacker"
    approval_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromotionBlocked, match="digest"):
        load_promotion_approval(approval_path)
