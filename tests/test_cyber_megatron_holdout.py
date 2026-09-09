import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.cyber_megatron_candidate import build_cyber_megatron_candidate
from koschei_sentinel.cyber_megatron_holdout import (
    build_cyber_megatron_holdout_plan,
    verify_cyber_megatron_holdout_plan,
)
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutInferenceCase,
    GoldHoldoutInferenceManifest,
)
from koschei_sentinel.gold_holdout_pack_signing import sign_gold_holdout_inference_pack
from koschei_sentinel.gold_reviewer_trust import (
    build_gold_reviewer_trust_policy,
    write_gold_reviewer_trust_policy,
)
from koschei_sentinel.training import canonical_json
from tests.test_cyber_megatron_candidate import _fixture, _relative

_GOLD_AUDIT_SHA = "5" * 64
_REVIEW_AUDIT_SHA = "d" * 64


def _write_public_key(path, private_key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _signed_owner_trusted_pack(tmp_path, *, gold_audit_sha: str = _GOLD_AUDIT_SHA):
    pack = tmp_path / "397b-holdout-pack"
    pack.mkdir()
    context = {
        "scenario_id": "scenario-397b-001",
        "critical_entity_ids": ["asset-001"],
        "graph_snapshots": [],
    }
    context_sha = hashlib.sha256(canonical_json(context).encode("utf-8")).hexdigest()
    case = GoldHoldoutInferenceCase(
        case_id="holdout-397b-001",
        scenario_id="scenario-397b-001",
        input_context=context,
        input_context_sha256=context_sha,
    )
    inputs = (
        json.dumps(
            case.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    (pack / "inputs.jsonl").write_text(inputs, encoding="utf-8")
    manifest = GoldHoldoutInferenceManifest(
        case_count=1,
        case_ids=[case.case_id],
        inputs_sha256=hashlib.sha256(inputs.encode("utf-8")).hexdigest(),
        source_gold_audit_sha256=gold_audit_sha,
    )
    manifest_path = pack / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reviewer = Ed25519PrivateKey.generate()
    proof = sign_gold_holdout_inference_pack(
        manifest_path,
        reviewer,
        review_signature_audit_sha256=_REVIEW_AUDIT_SHA,
    )
    signature_path = tmp_path / "397b-holdout-pack-signature.json"
    signature_path.write_text(
        json.dumps(proof.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reviewer_public_key_path = tmp_path / "397b-reviewer-public.pem"
    _write_public_key(reviewer_public_key_path, reviewer)

    owner = Ed25519PrivateKey.generate()
    policy = build_gold_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="gold-reviewer-397b-v1",
    )
    trust_policy_path = tmp_path / "397b-reviewer-trust.json"
    write_gold_reviewer_trust_policy(policy, trust_policy_path)
    owner_public_key_path = tmp_path / "397b-owner-public.pem"
    _write_public_key(owner_public_key_path, owner)
    return (
        pack,
        signature_path,
        reviewer_public_key_path,
        trust_policy_path,
        owner_public_key_path,
    )


def _candidate(tmp_path):
    _config_obj, config_path, _dataset, _identity, _plan, plan_path, checkpoint = _fixture(
        tmp_path
    )
    candidate_path = tmp_path / "build" / "candidates" / "397b.json"
    candidate = build_cyber_megatron_candidate(
        root=tmp_path,
        config_path=_relative(tmp_path, config_path),
        plan_path=_relative(tmp_path, plan_path),
        checkpoint_dir=_relative(tmp_path, checkpoint),
        output_path=_relative(tmp_path, candidate_path),
    )
    return candidate, candidate_path, config_path, plan_path, checkpoint


def _plan_kwargs(tmp_path):
    candidate, candidate_path, config_path, training_plan_path, checkpoint = _candidate(
        tmp_path
    )
    pack, signature, reviewer_key, trust_policy, owner_key = _signed_owner_trusted_pack(
        tmp_path
    )
    return candidate, {
        "root": tmp_path,
        "candidate_manifest_path": _relative(tmp_path, candidate_path),
        "candidate_config_path": _relative(tmp_path, config_path),
        "candidate_training_plan_path": _relative(tmp_path, training_plan_path),
        "checkpoint_dir": _relative(tmp_path, checkpoint),
        "inference_pack": pack,
        "signature_path": signature,
        "reviewer_public_key_path": reviewer_key,
        "reviewer_trust_policy_path": trust_policy,
        "owner_public_key_path": owner_key,
        "minimum_case_count": 1,
    }


def test_397b_holdout_plan_binds_candidate_signed_pack_and_owner_root(tmp_path) -> None:
    candidate, kwargs = _plan_kwargs(tmp_path)
    plan_path = tmp_path / "build" / "holdout" / "397b-plan.json"

    plan = build_cyber_megatron_holdout_plan(
        **kwargs,
        output_path=_relative(tmp_path, plan_path),
    )

    assert plan.model == candidate.model
    assert plan.model_revision == candidate.model_revision
    assert plan.candidate_sha256 == candidate.candidate_sha256
    assert plan.checkpoint_tree_sha256 == candidate.checkpoint_tree_sha256
    assert plan.gold_release_audit_sha256 == _GOLD_AUDIT_SHA
    assert plan.prediction_identity_digest == candidate.checkpoint_tree_sha256
    assert plan.answer_key_isolated is True
    assert plan.deterministic_generation is True
    assert plan.minimum_case_count == 1
    assert plan.case_count == 1

    verification = verify_cyber_megatron_holdout_plan(
        **kwargs,
        plan_path=_relative(tmp_path, plan_path),
    )
    assert verification.valid is True
    assert verification.plan == plan
    assert verification.violations == []


def test_397b_holdout_production_floor_rejects_tiny_pack(tmp_path) -> None:
    _candidate_obj, kwargs = _plan_kwargs(tmp_path)
    kwargs.pop("minimum_case_count")

    with pytest.raises(ValueError, match="below minimum case count"):
        build_cyber_megatron_holdout_plan(
            **kwargs,
            output_path="build/holdout/397b-plan.json",
        )


def test_397b_holdout_rejects_pack_from_different_gold_audit(tmp_path) -> None:
    _candidate_obj, candidate_path, config_path, training_plan_path, checkpoint = _candidate(
        tmp_path
    )
    pack, signature, reviewer_key, trust_policy, owner_key = _signed_owner_trusted_pack(
        tmp_path,
        gold_audit_sha="a" * 64,
    )

    with pytest.raises(ValueError, match="candidate Gold audit differs"):
        build_cyber_megatron_holdout_plan(
            root=tmp_path,
            candidate_manifest_path=_relative(tmp_path, candidate_path),
            candidate_config_path=_relative(tmp_path, config_path),
            candidate_training_plan_path=_relative(tmp_path, training_plan_path),
            checkpoint_dir=_relative(tmp_path, checkpoint),
            inference_pack=pack,
            signature_path=signature,
            reviewer_public_key_path=reviewer_key,
            reviewer_trust_policy_path=trust_policy,
            owner_public_key_path=owner_key,
            minimum_case_count=1,
            output_path="build/holdout/397b-plan.json",
        )


def test_397b_holdout_rejects_attacker_owner_root(tmp_path) -> None:
    _candidate_obj, kwargs = _plan_kwargs(tmp_path)
    attacker = Ed25519PrivateKey.generate()
    attacker_path = tmp_path / "attacker-owner-public.pem"
    _write_public_key(attacker_path, attacker)
    kwargs["owner_public_key_path"] = attacker_path

    with pytest.raises(ValueError, match="owner public key does not match"):
        build_cyber_megatron_holdout_plan(
            **kwargs,
            output_path="build/holdout/397b-plan.json",
        )


def test_397b_holdout_plan_verification_detects_checkpoint_mutation(tmp_path) -> None:
    _candidate_obj, kwargs = _plan_kwargs(tmp_path)
    plan_path = tmp_path / "build" / "holdout" / "397b-plan.json"
    build_cyber_megatron_holdout_plan(
        **kwargs,
        output_path=_relative(tmp_path, plan_path),
    )

    checkpoint = tmp_path / str(kwargs["checkpoint_dir"])
    (checkpoint / "model.safetensors").write_bytes(b"tampered-after-plan")
    verification = verify_cyber_megatron_holdout_plan(
        **kwargs,
        plan_path=_relative(tmp_path, plan_path),
    )

    assert verification.valid is False
    assert any("source revalidation failed" in row for row in verification.violations)
