from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.candidate_finalization import CandidateFinalization
from koschei_sentinel.launch_readiness import audit_launch_readiness
from koschei_sentinel.production_authority import (
    approve_production_authority,
    build_production_authority_proposal,
)


DIGEST = "a" * 64


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_model(path: Path, model) -> Path:
    path.write_text(json.dumps(model.model_dump(mode="json")), encoding="utf-8")
    return path


def _finalization() -> dict:
    return {
        "schema_version": "sentinel.candidate-finalization.v1",
        "candidate_id": "candidate-1",
        "state": "finalized_incubation",
        "authority": "explanation_only",
        "offline_receipt_digest": DIGEST,
        "job_digest": DIGEST,
        "adapter_manifest_digest": DIGEST,
        "adapter_digest": DIGEST,
        "dataset_manifest_digest": DIGEST,
        "training_config_digest": DIGEST,
        "comparison_digest": DIGEST,
        "benchmark_suite_digest": DIGEST,
        "benchmark_report_digest": DIGEST,
        "candidate_record_digest": DIGEST,
        "previous_registry_digest": DIGEST,
        "updated_registry_digest": DIGEST,
        "automatic_registry_replacement_allowed": False,
        "automatic_promotion_allowed": False,
        "production_deployment_allowed": False,
        "finalization_digest": DIGEST,
    }


def _owner_keypair(tmp_path: Path):
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    public_path = tmp_path / "owner.pub.pem"
    public_path.write_bytes(
        public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private, public, public_path


def test_launch_readiness_requires_verifiable_production_authority(tmp_path: Path) -> None:
    finalization = _write(tmp_path / "candidate-finalization.json", _finalization())
    holdout = _write(
        tmp_path / "holdout.json",
        {"schema_version": "sentinel.gold-holdout-evaluation.v1", "ok": True},
    )
    authority = _write(
        tmp_path / "authority.json",
        {
            "candidate_id": "candidate-1",
            "production_deployment_allowed": True,
        },
    )

    report = audit_launch_readiness(
        finalization_path=finalization,
        holdout_evidence_paths=[holdout],
        production_authority_path=authority,
    )

    assert report.market_release_ready is False
    assert report.production_authority_present is True
    assert report.production_authority_verified is False
    assert any("proposal is required" in item for item in report.blockers)


def test_launch_readiness_accepts_signed_canary_authority(tmp_path: Path) -> None:
    finalization_payload = _finalization()
    finalization = _write(tmp_path / "candidate-finalization.json", finalization_payload)
    holdout = _write(
        tmp_path / "holdout.json",
        {"schema_version": "sentinel.gold-holdout-evaluation.v1", "ok": True},
    )
    private, public, public_path = _owner_keypair(tmp_path)
    finalization_model = CandidateFinalization.model_validate(finalization_payload)
    proposal = build_production_authority_proposal(
        finalization_model,
        [holdout],
        public,
        max_initial_traffic_percent=5,
    )
    authority = approve_production_authority(
        proposal,
        private,
        approver_id="owner-1",
    )
    proposal_path = _write_model(tmp_path / "production-proposal.json", proposal)
    authority_path = _write_model(tmp_path / "production-authority.json", authority)

    report = audit_launch_readiness(
        finalization_path=finalization,
        holdout_evidence_paths=[holdout],
        production_authority_path=authority_path,
        production_authority_proposal_path=proposal_path,
        owner_public_key_path=public_path,
    )

    assert report.market_release_ready is True
    assert report.production_authority_verified is True
    assert report.deployment_scope == "canary_only"
    assert report.max_initial_traffic_percent == 5
    assert report.blockers == []


def test_launch_readiness_blocks_missing_holdout_and_authority(tmp_path: Path) -> None:
    finalization = _write(tmp_path / "candidate-finalization.json", _finalization())

    report = audit_launch_readiness(
        finalization_path=finalization,
        holdout_evidence_paths=[],
    )

    assert report.market_release_ready is False
    assert report.holdout_evidence_count == 0
    assert any("no independent holdout evidence" in item for item in report.blockers)
    assert any("no explicit signed production" in item for item in report.blockers)
