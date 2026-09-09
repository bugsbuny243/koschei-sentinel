import pytest

from koschei_sentinel.cyber_megatron_gold_evidence import (
    CyberMegatronGoldEvaluationEvidence,
    build_cyber_megatron_gold_evidence,
    verify_cyber_megatron_gold_evidence,
)
from koschei_sentinel.gold_holdout_evaluation import GoldHoldoutEvaluationPolicy
from tests.test_cyber_defense_promotion_v5 import _gold_evidence, _sha


def test_397b_gold_evidence_self_hash_and_report_identity_verify() -> None:
    evidence = _gold_evidence()

    verify_cyber_megatron_gold_evidence(evidence)
    assert len(evidence.model_revision) == 40
    assert len(evidence.checkpoint_tree_sha256) == 64
    assert evidence.report.adapter_digest == evidence.checkpoint_tree_sha256


def test_397b_gold_evidence_rejects_checkpoint_report_identity_split() -> None:
    evidence = _gold_evidence()
    payload = evidence.model_dump(mode="json")
    payload.pop("evidence_sha256")
    payload["checkpoint_tree_sha256"] = "8" * 64
    payload["evidence_sha256"] = _sha(payload)

    with pytest.raises(ValueError, match="report identity"):
        CyberMegatronGoldEvaluationEvidence.model_validate(payload)


def test_397b_gold_evidence_cannot_lower_production_case_floor(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 50"):
        build_cyber_megatron_gold_evidence(
            root=tmp_path,
            release_dir=tmp_path / "release",
            worker_dir="worker",
            holdout_plan_path="holdout.json",
            candidate_manifest_path="candidate.json",
            candidate_config_path="config.json",
            candidate_training_plan_path="training-plan.json",
            checkpoint_dir="checkpoint-100-merged",
            inference_pack=tmp_path / "pack",
            signature_path=tmp_path / "pack-signature.json",
            reviewer_public_key_path=tmp_path / "reviewer.pem",
            reviewer_trust_policy_path=tmp_path / "reviewer-trust.json",
            owner_public_key_path=tmp_path / "owner.pem",
            policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1),
            minimum_case_count=1,
        )
