import hashlib

import pytest

from koschei_sentinel.cyber_defense_promotion_v5 import (
    build_cyber_defense_promotion_v5,
)
from koschei_sentinel.cyber_megatron_gold_evidence import (
    CyberMegatronGoldEvaluationEvidence,
)
from koschei_sentinel.cyber_megatron_training import (
    QWEN35_397B_MODEL,
    QWEN35_397B_REVISION,
)
from koschei_sentinel.cyber_training_bundle import CyberTrainingBundle, CyberTrainingStage
from koschei_sentinel.gold_holdout_evaluation import (
    GoldHoldoutEvaluationPolicy,
    GoldHoldoutEvaluationReport,
)
from koschei_sentinel.training import canonical_json
from tests.test_cyber_defense_promotion import _load, _multi, _single

_CHECKPOINT_SHA = "9" * 64


def _sha(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _bundle_digest(payload: dict[str, object]) -> str:
    value = "|".join(
        [
            str(payload["bundle_id"]),
            str(payload["foundation_model_ref"]),
            str(payload["foundation_model_revision"]),
            str(payload["knowledge_batch_id"]),
            str(payload["knowledge_training_corpus_sha256"]),
            str(payload["knowledge_artifact_manifest_sha256"]),
            str(payload["defense_reflex_examples_sha256"]),
            str(payload["defense_reflex_example_count"]),
            str(payload["causal_defense_examples_sha256"]),
            str(payload["causal_defense_example_count"]),
            str(payload["eval_holdout_sha256"]),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bundle() -> CyberTrainingBundle:
    payload = {
        "bundle_id": "bundle:397b-promotion-v5",
        "foundation_model_ref": QWEN35_397B_MODEL,
        "foundation_model_revision": QWEN35_397B_REVISION,
        "knowledge_batch_id": "batch:397b",
        "knowledge_training_corpus_sha256": "a" * 64,
        "knowledge_artifact_manifest_sha256": "b" * 64,
        "defense_reflex_examples_sha256": "c" * 64,
        "defense_reflex_example_count": 100,
        "causal_defense_examples_sha256": "d" * 64,
        "causal_defense_example_count": 100,
        "eval_holdout_sha256": "e" * 64,
        "required_stages": [
            CyberTrainingStage.KNOWLEDGE_CONTINUED_PRETRAINING,
            CyberTrainingStage.DEFENSE_REFLEX_SFT,
            CyberTrainingStage.ADVERSARIAL_REASONING,
            CyberTrainingStage.CYBER_RANGE_REGRESSION,
        ],
        "ready_for_training": True,
    }
    return CyberTrainingBundle(
        **payload,
        bundle_sha256=_bundle_digest(payload),
    )


def _policy() -> GoldHoldoutEvaluationPolicy:
    return GoldHoldoutEvaluationPolicy(minimum_case_count=50)


def _report(*, passed: bool = True) -> GoldHoldoutEvaluationReport:
    metric = 1.0 if passed else 0.5
    payload = {
        "schema_version": "sentinel.gold-holdout-evaluation-report.v2",
        "model_ref": QWEN35_397B_MODEL,
        "model_revision": QWEN35_397B_REVISION,
        "adapter_digest": _CHECKPOINT_SHA,
        "case_count": 50,
        "prediction_count": 50,
        "missing_case_ids": [],
        "extra_case_ids": [],
        "structural_exact_cases": 50 if passed else 25,
        "structural_exact_rate": metric,
        "compared_steps": 50,
        "predicted_steps": 50,
        "mode_accuracy": metric,
        "action_accuracy": metric,
        "target_accuracy": metric,
        "evidence_selection_accuracy": metric,
        "evidence_grounding_rate": 1.0,
        "target_grounding_rate": 1.0,
        "outcome_verification_rate": 1.0,
        "case_results": [],
        "passed": passed,
        "violations": [] if passed else ["fixture Gold HOLDOUT failure"],
    }
    return GoldHoldoutEvaluationReport(
        **payload,
        report_sha256=_sha(payload),
    )


def _gold_evidence(*, passed: bool = True) -> CyberMegatronGoldEvaluationEvidence:
    policy = _policy()
    report = _report(passed=passed)
    payload = {
        "schema_version": "sentinel.cyber-megatron-gold-evaluation-evidence.v1",
        "backend": "megatron-swift-vllm",
        "model": QWEN35_397B_MODEL,
        "model_revision": QWEN35_397B_REVISION,
        "run_id": "qwen3p5-397b-gold-production",
        "candidate_manifest_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "checkpoint_tree_sha256": _CHECKPOINT_SHA,
        "source_gold_audit_sha256": "3" * 64,
        "review_signature_audit_sha256": "4" * 64,
        "pack_signature_file_sha256": "5" * 64,
        "pack_proof_sha256": "6" * 64,
        "reviewer_trust_policy_file_sha256": "7" * 64,
        "reviewer_trust_policy_digest": "8" * 64,
        "owner_key_fingerprint": "a" * 64,
        "holdout_plan_file_sha256": "b" * 64,
        "holdout_plan_sha256": "c" * 64,
        "worker_plan_sha256": "d" * 64,
        "execution_state_sha256": "e" * 64,
        "inference_receipt_sha256": "f" * 64,
        "output_verification_sha256": "0" * 64,
        "generation_policy_sha256": "1" * 64,
        "evaluation_policy_sha256": _sha(policy.model_dump(mode="json")),
        "minimum_case_count": 50,
        "case_count": 50,
        "prediction_count": 50,
        "failure_count": 0,
        "output_verification_valid": True,
        "complete_case_accounting": True,
        "report": report.model_dump(mode="json"),
        "passed": passed,
    }
    return CyberMegatronGoldEvaluationEvidence(
        **payload,
        evidence_sha256=_sha(payload),
    )


def _promotion(**overrides):
    kwargs = {
        "promotion_id": "promotion:397b-v5",
        "candidate_model_ref": QWEN35_397B_MODEL,
        "candidate_model_revision": QWEN35_397B_REVISION,
        "candidate_checkpoint_sha256": _CHECKPOINT_SHA,
        "training_bundle": _bundle(),
        "cyber_range_report": _single(),
        "multi_incident_range_report": _multi(),
        "defense_load_range_report": _load(),
        "gold_evidence": _gold_evidence(),
        "gold_policy": _policy(),
    }
    kwargs.update(overrides)
    return build_cyber_defense_promotion_v5(**kwargs)


def test_promotion_v5_separates_model_revision_from_checkpoint_identity() -> None:
    promotion = _promotion()

    assert promotion.candidate_model_revision == QWEN35_397B_REVISION
    assert len(promotion.candidate_model_revision) == 40
    assert promotion.candidate_checkpoint_sha256 == _CHECKPOINT_SHA
    assert len(promotion.candidate_checkpoint_sha256) == 64
    assert promotion.candidate_model_revision != promotion.candidate_checkpoint_sha256
    assert promotion.ready_for_promotion is True


def test_promotion_v5_rejects_legacy_digest_as_model_revision() -> None:
    with pytest.raises(ValueError, match="pinned model revision"):
        _promotion(candidate_model_revision=_CHECKPOINT_SHA)


def test_promotion_v5_rejects_checkpoint_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="checkpoint differs"):
        _promotion(candidate_checkpoint_sha256="8" * 64)


def test_promotion_v5_rejects_gold_policy_below_production_floor() -> None:
    with pytest.raises(ValueError, match="at least 50"):
        _promotion(gold_policy=GoldHoldoutEvaluationPolicy(minimum_case_count=1))


def test_promotion_v5_rejects_forged_training_bundle_digest() -> None:
    forged = _bundle().model_copy(update={"bundle_sha256": "0" * 64})

    with pytest.raises(ValueError, match="bundle self-binding digest"):
        _promotion(training_bundle=forged)


def test_promotion_v5_fails_closed_when_gold_gate_fails() -> None:
    promotion = _promotion(gold_evidence=_gold_evidence(passed=False))

    assert promotion.gold_holdout_passed is False
    assert promotion.ready_for_promotion is False
