from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.candidate_finalization import CandidateFinalization
from koschei_sentinel.promotion import (
    approve_promotion_proposal,
    build_promotion_proposal,
)
from koschei_sentinel.shadow_replay import (
    ShadowReplayBlocked,
    build_shadow_replay_plan,
    load_shadow_replay_plan,
    write_shadow_replay_plan,
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


def _approval(tmp_path: Path):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_path = tmp_path / "owner-public.pem"
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    proposal = build_promotion_proposal(_finalization(), public_key)
    approval = approve_promotion_proposal(
        proposal,
        private_key,
        approver_id="owner@koschei",
    )
    return proposal, approval, public_path


def test_signed_approval_creates_only_an_offline_shadow_plan(tmp_path: Path) -> None:
    proposal, approval, public_path = _approval(tmp_path)
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        '{"case_id":"case-1","evidence":[]}\n'
        '{"case_id":"case-2","evidence":[]}\n',
        encoding="utf-8",
    )

    plan = build_shadow_replay_plan(
        proposal,
        approval,
        owner_public_key_path=public_path,
        replay_path=replay,
        output_dir=tmp_path / "shadow-output",
        root=tmp_path,
    )

    assert plan.stage == "shadow_research_candidate"
    assert plan.replay_cases == 2
    assert plan.manual_dispatch_required is True
    assert plan.network_access_allowed is False
    assert plan.live_chain_reads_allowed is False
    assert plan.live_customer_traffic_allowed is False
    assert plan.verdict_mutation_allowed is False
    assert plan.production_deployment_allowed is False
    assert plan.web3_runtime_integration_allowed is False


def test_wrong_owner_key_blocks_shadow_plan(tmp_path: Path) -> None:
    proposal, approval, _ = _approval(tmp_path)
    wrong_public = Ed25519PrivateKey.generate().public_key()
    wrong_path = tmp_path / "wrong-owner.pem"
    wrong_path.write_bytes(
        wrong_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"case_id":"case-1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="public key"):
        build_shadow_replay_plan(
            proposal,
            approval,
            owner_public_key_path=wrong_path,
            replay_path=replay,
            output_dir=tmp_path / "output",
            root=tmp_path,
        )


def test_duplicate_replay_identifiers_fail_closed(tmp_path: Path) -> None:
    proposal, approval, public_path = _approval(tmp_path)
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        '{"case_id":"same"}\n{"case_id":"same"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ShadowReplayBlocked, match="duplicate"):
        build_shadow_replay_plan(
            proposal,
            approval,
            owner_public_key_path=public_path,
            replay_path=replay,
            output_dir=tmp_path / "output",
            root=tmp_path,
        )


def test_plan_digest_tampering_and_overwrite_are_rejected(tmp_path: Path) -> None:
    proposal, approval, public_path = _approval(tmp_path)
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"test_id":"test-1"}\n', encoding="utf-8")
    plan = build_shadow_replay_plan(
        proposal,
        approval,
        owner_public_key_path=public_path,
        replay_path=replay,
        output_dir=tmp_path / "output",
        root=tmp_path,
    )
    path = tmp_path / "shadow-plan.json"
    write_shadow_replay_plan(plan, path)

    with pytest.raises(FileExistsError):
        write_shadow_replay_plan(plan, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["network_access_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid shadow replay plan"):
        load_shadow_replay_plan(path)
