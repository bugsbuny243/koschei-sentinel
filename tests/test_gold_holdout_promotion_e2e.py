from koschei_sentinel.cyber_defense_promotion import (
    build_cyber_defense_promotion_evidence_from_sources,
)
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    build_gold_holdout_evaluation_evidence,
)
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from tests.gold_candidate_binding_helpers import (
    rebind_inference_fixture_to_gold_candidate,
)
from tests.gold_review_signing_helpers import attach_signed_review_proofs
from tests.test_cyber_defense_promotion import _bundle, _load, _multi, _single
from tests.test_gold_holdout_evaluation_evidence import _passing_fixture


def test_gold_holdout_to_promotion_v4_end_to_end(tmp_path, monkeypatch) -> None:
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
    reviewer_public_key = attach_signed_review_proofs(release)

    verification = verify_gold_holdout_inference_output(
        output,
        pack,
        candidate_export,
    )
    assert verification.valid is True
    assert verification.complete_case_accounting is True
    assert verification.failure_count == 0

    policy = GoldHoldoutEvaluationPolicy(minimum_case_count=1)
    evidence = build_gold_holdout_evaluation_evidence(
        release_dir=release,
        inference_pack_dir=pack,
        inference_output_dir=output,
        candidate_export_dir=candidate_export,
        policy=policy,
        reviewer_public_key=reviewer_public_key,
    )
    assert evidence.passed is True
    assert evidence.review_signature_audit_sha256 is not None
    assert evidence.candidate_training_binding_verification_sha256 is not None
    assert evidence.inference_verification_sha256 == verification.verification_sha256
    assert evidence.inference_plan_sha256 == plan.plan_sha256
    assert evidence.adapter_digest == plan.adapter_digest

    promotion = build_cyber_defense_promotion_evidence_from_sources(
        promotion_id="promotion:gold-holdout-e2e",
        candidate_model_ref=plan.model_ref,
        candidate_model_revision=plan.adapter_digest,
        training_bundle=_bundle(),
        cyber_range_report=_single(),
        multi_incident_range_report=_multi(),
        defense_load_range_report=_load(),
        supplied_gold_holdout_evidence=evidence,
        gold_holdout_policy=policy,
        gold_release_dir=release,
        gold_inference_pack_dir=pack,
        gold_inference_output_dir=output,
        gold_candidate_export_dir=candidate_export,
        gold_reviewer_public_key=reviewer_public_key,
    )

    assert promotion.schema_version == "sentinel.cyber-defense-promotion-evidence.v4"
    assert promotion.gold_holdout_passed is True
    assert promotion.ready_for_promotion is True
    assert promotion.candidate_model_revision == plan.adapter_digest
    assert promotion.gold_review_signature_audit_sha256 == evidence.review_signature_audit_sha256
    assert (
        promotion.gold_candidate_training_binding_sha256
        == evidence.candidate_training_binding_verification_sha256
    )
    assert promotion.gold_holdout_evaluation_evidence_sha256 == evidence.evidence_sha256
    assert promotion.gold_holdout_evaluation_report_sha256 == evidence.report.report_sha256
    assert (
        promotion.gold_holdout_inference_verification_sha256
        == verification.verification_sha256
    )
