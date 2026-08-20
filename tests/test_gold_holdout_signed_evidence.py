import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import koschei_sentinel.gold_holdout_evaluation_evidence as evidence_module
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    build_gold_holdout_evaluation_evidence,
    build_owner_trusted_gold_holdout_evaluation_evidence,
)
from koschei_sentinel.gold_holdout_pack_signing import (
    sign_gold_holdout_inference_pack,
)
from koschei_sentinel.gold_reviewer_trust import build_gold_reviewer_trust_policy
from koschei_sentinel.gold_review_signing import audit_gold_release_review_signatures
from tests.gold_candidate_binding_helpers import (
    rebind_inference_fixture_to_gold_candidate,
)
from tests.gold_review_signing_helpers import attach_signed_review_proofs
from tests.test_gold_holdout_evaluation_evidence import _passing_fixture


def _gold_bound_fixture(tmp_path, monkeypatch):
    release, pack, output, plan, _candidate_export = _passing_fixture(
        tmp_path,
        monkeypatch,
    )
    plan, candidate_export = rebind_inference_fixture_to_gold_candidate(
        tmp_path,
        monkeypatch,
        release=release,
        pack=pack,
        output=output,
        model_ref=plan.model_ref,
    )
    return release, pack, output, plan, candidate_export


def _signed_release_and_pack(release, pack):
    reviewer_private_key = attach_signed_review_proofs(
        release,
        return_private_key=True,
    )
    signature_audit = audit_gold_release_review_signatures(
        release,
        reviewer_private_key.public_key(),
    )
    assert signature_audit.valid is True
    pack_proof = sign_gold_holdout_inference_pack(
        pack / "manifest.json",
        reviewer_private_key,
        review_signature_audit_sha256=signature_audit.audit_sha256,
    )
    return reviewer_private_key, pack_proof


def test_owner_trusted_evidence_rejects_wrong_owner_before_artifact_rebuild(
    monkeypatch,
) -> None:
    reviewer = Ed25519PrivateKey.generate()
    owner = Ed25519PrivateKey.generate()
    trust_policy = build_gold_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="gold-reviewer-v1",
    )
    rebuilt = False

    def forbidden_rebuild(**_kwargs):
        nonlocal rebuilt
        rebuilt = True
        raise AssertionError("artifact rebuild must not start under an untrusted owner root")

    monkeypatch.setattr(
        evidence_module,
        "build_gold_holdout_evaluation_evidence",
        forbidden_rebuild,
    )

    with pytest.raises(ValueError, match="owner public key does not match"):
        build_owner_trusted_gold_holdout_evaluation_evidence(
            release_dir="release",
            inference_pack_dir="pack",
            inference_output_dir="output",
            candidate_export_dir="candidate",
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            reviewer_public_key=reviewer.public_key(),
            reviewer_trust_policy=trust_policy,
            owner_public_key=Ed25519PrivateKey.generate().public_key(),
            inference_pack_signature_proof=object(),
        )

    assert rebuilt is False


def test_owner_trust_policy_digest_changes_gold_evidence_identity(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _gold_bound_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_private_key, pack_proof = _signed_release_and_pack(release, pack)
    owner_private_key = Ed25519PrivateKey.generate()
    first_policy = build_gold_reviewer_trust_policy(
        reviewer_private_key.public_key(),
        owner_private_key,
        policy_id="gold-reviewer-v1",
    )
    second_policy = build_gold_reviewer_trust_policy(
        reviewer_private_key.public_key(),
        owner_private_key,
        policy_id="gold-reviewer-v2",
    )
    evaluation_policy = GoldHoldoutEvaluationPolicy(minimum_case_count=1)
    common = {
        "release_dir": release,
        "inference_pack_dir": pack,
        "inference_output_dir": output,
        "candidate_export_dir": candidate_export,
        "policy": evaluation_policy,
        "reviewer_public_key": reviewer_private_key.public_key(),
        "owner_public_key": owner_private_key.public_key(),
        "inference_pack_signature_proof": pack_proof,
    }

    first = build_owner_trusted_gold_holdout_evaluation_evidence(
        **common,
        reviewer_trust_policy=first_policy,
    )
    second = build_owner_trusted_gold_holdout_evaluation_evidence(
        **common,
        reviewer_trust_policy=second_policy,
    )

    assert first.reviewer_trust_policy_sha256 == first_policy.policy_digest
    assert second.reviewer_trust_policy_sha256 == second_policy.policy_digest
    assert first.owner_key_fingerprint == second.owner_key_fingerprint
    assert first.evidence_sha256 != second.evidence_sha256


def test_gold_evidence_binds_valid_reviewer_signature_and_training_audits(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _gold_bound_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_private_key, pack_proof = _signed_release_and_pack(release, pack)

    evidence = build_gold_holdout_evaluation_evidence(
        release_dir=release,
        inference_pack_dir=pack,
        inference_output_dir=output,
        candidate_export_dir=candidate_export,
        policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
        reviewer_public_key=reviewer_private_key.public_key(),
        inference_pack_signature_proof=pack_proof,
    )

    assert evidence.passed is True
    assert evidence.review_signature_audit_sha256 is not None
    assert evidence.review_signature_audit_sha256 == pack_proof.review_signature_audit_sha256
    assert evidence.candidate_training_binding_verification_sha256 is not None
    assert evidence.inference_pack_signature_proof_sha256 == pack_proof.proof_sha256
    assert evidence.reviewer_trust_policy_sha256 is None
    assert evidence.owner_key_fingerprint is None


def test_gold_evidence_rejects_untrusted_reviewer_key(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _gold_bound_fixture(
        tmp_path,
        monkeypatch,
    )
    _reviewer_private_key, pack_proof = _signed_release_and_pack(release, pack)
    wrong_key = Ed25519PrivateKey.generate().public_key()

    with pytest.raises(ValueError, match="untrusted reviewer key"):
        build_gold_holdout_evaluation_evidence(
            release_dir=release,
            inference_pack_dir=pack,
            inference_output_dir=output,
            candidate_export_dir=candidate_export,
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            reviewer_public_key=wrong_key,
            inference_pack_signature_proof=pack_proof,
        )


def test_gold_evidence_rejects_missing_pack_signature_in_production_path(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _gold_bound_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_private_key = attach_signed_review_proofs(
        release,
        return_private_key=True,
    )

    with pytest.raises(ValueError, match="requires a signed inference-pack proof"):
        build_gold_holdout_evaluation_evidence(
            release_dir=release,
            inference_pack_dir=pack,
            inference_output_dir=output,
            candidate_export_dir=candidate_export,
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            reviewer_public_key=reviewer_private_key.public_key(),
        )


def test_gold_evidence_rejects_pack_signed_against_different_review_audit(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _gold_bound_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_private_key = attach_signed_review_proofs(
        release,
        return_private_key=True,
    )
    wrong_audit_proof = sign_gold_holdout_inference_pack(
        pack / "manifest.json",
        reviewer_private_key,
        review_signature_audit_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="different signed-review audit"):
        build_gold_holdout_evaluation_evidence(
            release_dir=release,
            inference_pack_dir=pack,
            inference_output_dir=output,
            candidate_export_dir=candidate_export,
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            reviewer_public_key=reviewer_private_key.public_key(),
            inference_pack_signature_proof=wrong_audit_proof,
        )


def test_gold_evidence_rejects_signed_release_with_unrelated_candidate_corpus(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _passing_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_private_key, pack_proof = _signed_release_and_pack(release, pack)

    with pytest.raises(ValueError, match="candidate training binding failed"):
        build_gold_holdout_evaluation_evidence(
            release_dir=release,
            inference_pack_dir=pack,
            inference_output_dir=output,
            candidate_export_dir=candidate_export,
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            reviewer_public_key=reviewer_private_key.public_key(),
            inference_pack_signature_proof=pack_proof,
        )
