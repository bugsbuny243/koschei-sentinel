import shutil

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.cyber_defense_promotion import (
    build_cyber_defense_promotion_evidence_from_sources,
)
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.gold_holdout_evaluation_evidence import (
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
from tests.test_cyber_defense_promotion import _bundle, _load, _multi, _single
from tests.test_gold_holdout_evaluation_evidence import _passing_fixture


def test_source_reverified_promotion_survives_full_artifact_relocation(
    tmp_path,
    monkeypatch,
) -> None:
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
    reviewer_private_key = attach_signed_review_proofs(
        release,
        return_private_key=True,
    )
    reviewer_public_key = reviewer_private_key.public_key()
    owner_private_key = Ed25519PrivateKey.generate()
    reviewer_trust_policy = build_gold_reviewer_trust_policy(
        reviewer_public_key,
        owner_private_key,
        policy_id="gold-reviewer-v1",
    )
    signature_audit = audit_gold_release_review_signatures(
        release,
        reviewer_public_key,
    )
    assert signature_audit.valid is True
    pack_proof = sign_gold_holdout_inference_pack(
        pack / "manifest.json",
        reviewer_private_key,
        review_signature_audit_sha256=signature_audit.audit_sha256,
    )
    policy = GoldHoldoutEvaluationPolicy(minimum_case_count=1)
    supplied = build_owner_trusted_gold_holdout_evaluation_evidence(
        release_dir=release,
        inference_pack_dir=pack,
        inference_output_dir=output,
        candidate_export_dir=candidate_export,
        policy=policy,
        reviewer_public_key=reviewer_public_key,
        reviewer_trust_policy=reviewer_trust_policy,
        owner_public_key=owner_private_key.public_key(),
        inference_pack_signature_proof=pack_proof,
    )

    relocated_root = tmp_path / "promotion-host"
    relocated_release = relocated_root / "gold-release"
    relocated_pack = relocated_root / "holdout-pack"
    relocated_output = relocated_root / "inference-output"
    relocated_candidate = relocated_root / "candidate-export"
    shutil.copytree(release, relocated_release)
    shutil.copytree(pack, relocated_pack)
    shutil.copytree(output, relocated_output)
    shutil.copytree(candidate_export, relocated_candidate)

    promotion = build_cyber_defense_promotion_evidence_from_sources(
        promotion_id="promotion:relocated-gold-holdout",
        candidate_model_ref=plan.model_ref,
        candidate_model_revision=plan.adapter_digest,
        training_bundle=_bundle(),
        cyber_range_report=_single(),
        multi_incident_range_report=_multi(),
        defense_load_range_report=_load(),
        supplied_gold_holdout_evidence=supplied,
        gold_holdout_policy=policy,
        gold_release_dir=relocated_release,
        gold_inference_pack_dir=relocated_pack,
        gold_inference_output_dir=relocated_output,
        gold_candidate_export_dir=relocated_candidate,
        gold_reviewer_public_key=reviewer_public_key,
        gold_reviewer_trust_policy=reviewer_trust_policy,
        gold_owner_public_key=owner_private_key.public_key(),
        gold_inference_pack_signature_proof=pack_proof,
    )

    assert promotion.ready_for_promotion is True
    assert promotion.gold_holdout_passed is True
    assert promotion.gold_review_signature_audit_sha256 == signature_audit.audit_sha256
    assert promotion.gold_candidate_training_binding_sha256 is not None
    assert promotion.gold_holdout_pack_signature_proof_sha256 == pack_proof.proof_sha256
    assert promotion.gold_holdout_evaluation_evidence_sha256 == supplied.evidence_sha256
    assert supplied.reviewer_trust_policy_sha256 == reviewer_trust_policy.policy_digest
    assert supplied.owner_key_fingerprint == reviewer_trust_policy.owner_key_fingerprint
    assert (
        promotion.gold_candidate_training_binding_sha256
        == supplied.candidate_training_binding_verification_sha256
    )
    assert (
        promotion.gold_holdout_pack_signature_proof_sha256
        == supplied.inference_pack_signature_proof_sha256
    )
    assert (
        promotion.gold_holdout_inference_verification_sha256
        == supplied.inference_verification_sha256
    )
