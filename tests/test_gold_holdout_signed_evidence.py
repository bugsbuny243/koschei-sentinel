import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    build_gold_holdout_evaluation_evidence,
)
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


def test_gold_evidence_binds_valid_reviewer_signature_and_training_audits(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _gold_bound_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_public_key = attach_signed_review_proofs(release)

    evidence = build_gold_holdout_evaluation_evidence(
        release_dir=release,
        inference_pack_dir=pack,
        inference_output_dir=output,
        candidate_export_dir=candidate_export,
        policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
        reviewer_public_key=reviewer_public_key,
    )

    assert evidence.passed is True
    assert evidence.review_signature_audit_sha256 is not None
    assert evidence.candidate_training_binding_verification_sha256 is not None


def test_gold_evidence_rejects_untrusted_reviewer_key(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _gold_bound_fixture(
        tmp_path,
        monkeypatch,
    )
    attach_signed_review_proofs(release)
    wrong_key = Ed25519PrivateKey.generate().public_key()

    with pytest.raises(ValueError, match="review signature audit failed"):
        build_gold_holdout_evaluation_evidence(
            release_dir=release,
            inference_pack_dir=pack,
            inference_output_dir=output,
            candidate_export_dir=candidate_export,
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            reviewer_public_key=wrong_key,
        )


def test_gold_evidence_rejects_signed_release_with_unrelated_candidate_corpus(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, _plan, candidate_export = _passing_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_public_key = attach_signed_review_proofs(release)

    with pytest.raises(ValueError, match="candidate training binding failed"):
        build_gold_holdout_evaluation_evidence(
            release_dir=release,
            inference_pack_dir=pack,
            inference_output_dir=output,
            candidate_export_dir=candidate_export,
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            reviewer_public_key=reviewer_public_key,
        )
