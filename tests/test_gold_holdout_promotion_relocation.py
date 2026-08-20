import shutil

from koschei_sentinel.cyber_defense_promotion import (
    build_cyber_defense_promotion_evidence_from_sources,
)
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from koschei_sentinel.gold_holdout_evaluation_evidence import (
    build_gold_holdout_evaluation_evidence,
)
from tests.gold_review_signing_helpers import attach_signed_review_proofs
from tests.test_cyber_defense_promotion import _bundle, _load, _multi, _single
from tests.test_gold_holdout_evaluation_evidence import _passing_fixture


def test_source_reverified_promotion_survives_full_artifact_relocation(
    tmp_path,
    monkeypatch,
) -> None:
    release, pack, output, plan, candidate_export = _passing_fixture(
        tmp_path,
        monkeypatch,
    )
    reviewer_public_key = attach_signed_review_proofs(release)
    policy = GoldHoldoutEvaluationPolicy(minimum_case_count=1)
    supplied = build_gold_holdout_evaluation_evidence(
        release_dir=release,
        inference_pack_dir=pack,
        inference_output_dir=output,
        candidate_export_dir=candidate_export,
        policy=policy,
        reviewer_public_key=reviewer_public_key,
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
    )

    assert promotion.ready_for_promotion is True
    assert promotion.gold_holdout_passed is True
    assert promotion.gold_review_signature_audit_sha256 is not None
    assert promotion.gold_holdout_evaluation_evidence_sha256 == supplied.evidence_sha256
    assert (
        promotion.gold_holdout_inference_verification_sha256
        == supplied.inference_verification_sha256
    )
