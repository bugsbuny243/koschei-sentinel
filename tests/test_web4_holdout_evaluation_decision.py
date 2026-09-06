from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.promotion import public_key_fingerprint
from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_holdout_evaluation import (
    Web4HoldoutCaseResult,
    Web4HoldoutEvaluationEvidence,
    Web4HoldoutEvaluationReport,
)
from koschei_sentinel.web4_holdout_evaluation_decision import (
    Web4EvaluationDecision,
    Web4HoldoutEvaluationDecision,
    Web4HoldoutEvaluationDecisionSpec,
    sign_web4_holdout_evaluation_decision,
    verify_web4_holdout_evaluation_decision,
)


def _digest(payload: object) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _evidence(
    owner: Ed25519PrivateKey,
    *,
    passed: bool = True,
    complete: bool = True,
    identity_suffix: str = "a",
) -> Web4HoldoutEvaluationEvidence:
    release_sha = identity_suffix * 64
    release_artifact_sha = "b" * 64
    inference_pack_sha = "c" * 64
    prediction_set_sha = "d" * 64
    evaluation_policy_sha = "e" * 64
    model_artifact_sha = "f" * 64
    missing = [] if complete else ["case-web4-001"]
    prediction_count = 1 if complete else 0
    exact_match_cases = 1 if passed and complete else 0
    exact_match_rate = 1.0 if passed and complete else 0.0
    matched_field_count = 3 if passed and complete else 2
    expected_field_accuracy = 1.0 if passed and complete else 2 / 3
    case_results = []
    if complete:
        case_results = [
            Web4HoldoutCaseResult(
                case_id="case-web4-001",
                family="delegation_scope_widening_detection",
                exact_match=passed,
                expected_field_count=3,
                matched_field_count=matched_field_count,
                missing_field_paths=[],
                mismatched_field_paths=[] if passed else ["authority_status"],
                unexpected_field_paths=[],
            )
        ]

    report_payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-evaluation-report.v1",
        "release_sha256": release_sha,
        "release_artifact_sha256": release_artifact_sha,
        "inference_pack_sha256": inference_pack_sha,
        "prediction_set_sha256": prediction_set_sha,
        "evaluation_policy_sha256": evaluation_policy_sha,
        "model_ref": "fixture/web4-owner-decision",
        "model_revision": "fixture-v1",
        "model_artifact_sha256": model_artifact_sha,
        "case_count": 1,
        "prediction_count": prediction_count,
        "answer_key_count": 1,
        "missing_case_ids": missing,
        "extra_case_ids": [],
        "exact_match_cases": exact_match_cases,
        "exact_match_rate": exact_match_rate,
        "expected_field_count": 3,
        "matched_field_count": matched_field_count,
        "expected_field_accuracy": expected_field_accuracy,
        "case_results": [row.model_dump(mode="json") for row in case_results],
        "answer_key_values_embedded": False,
        "complete_case_accounting": complete,
        "passed": passed,
        "violations": [] if passed else ["evaluation thresholds not met"],
    }
    report_payload["report_sha256"] = _digest(report_payload)
    report = Web4HoldoutEvaluationReport.model_validate(report_payload)

    evidence_payload: dict[str, object] = {
        "schema_version": "sentinel.web4-holdout-evaluation-evidence.v1",
        "release_id": f"web4-owner-decision-{identity_suffix}",
        "release_sha256": release_sha,
        "release_artifact_sha256": release_artifact_sha,
        "release_owner_key_fingerprint": public_key_fingerprint(owner.public_key()),
        "inference_pack_sha256": inference_pack_sha,
        "prediction_set_sha256": prediction_set_sha,
        "answer_key_bundle_sha256": "1" * 64,
        "evaluation_policy_sha256": evaluation_policy_sha,
        "model_ref": report.model_ref,
        "model_revision": report.model_revision,
        "model_artifact_sha256": model_artifact_sha,
        "case_count": 1,
        "prediction_count": prediction_count,
        "answer_key_count": 1,
        "complete_case_accounting": complete,
        "answer_key_values_embedded": False,
        "offline_replay": True,
        "network_access_required": False,
        "gpu_required": False,
        "research_evaluation_only": True,
        "training_authorization": False,
        "promotion_eligible": False,
        "production_activation_allowed": False,
        "report": report.model_dump(mode="json"),
        "passed": passed,
    }
    evidence_payload["evidence_sha256"] = _digest(evidence_payload)
    return Web4HoldoutEvaluationEvidence.model_validate(evidence_payload)


def _spec(decision: Web4EvaluationDecision) -> Web4HoldoutEvaluationDecisionSpec:
    return Web4HoldoutEvaluationDecisionSpec(
        decision_id="web4-owner-decision-001",
        approver_id="sentinel-owner",
        decision=decision,
        rationale="Owner reviewed the complete research evidence and recorded a bounded decision.",
    )


def test_owner_can_accept_only_passing_complete_research_evidence() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner)
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=_spec(Web4EvaluationDecision.ACCEPT),
        owner_private_key=owner,
    )

    assert decision.research_evaluation_evidence_accepted is True
    assert decision.research_comparison_eligible is True
    assert decision.research_evaluation_only is True
    assert decision.model_execution_authorized is False
    assert decision.training_authorization is False
    assert decision.promotion_eligible is False
    assert decision.production_activation_allowed is False
    assert decision.evidence_sha256 == evidence.evidence_sha256
    assert decision.owner_key_fingerprint == public_key_fingerprint(owner.public_key())


def test_accept_rejects_failed_evidence() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner, passed=False)

    with pytest.raises(ValueError, match="cannot accept failed"):
        sign_web4_holdout_evaluation_decision(
            evidence=evidence,
            spec=_spec(Web4EvaluationDecision.ACCEPT),
            owner_private_key=owner,
        )


def test_accept_rejects_incomplete_case_accounting() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner, passed=False, complete=False)

    with pytest.raises(ValueError, match="cannot accept"):
        sign_web4_holdout_evaluation_decision(
            evidence=evidence,
            spec=_spec(Web4EvaluationDecision.ACCEPT),
            owner_private_key=owner,
        )


def test_owner_can_reject_failed_evidence_without_granting_authority() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner, passed=False)
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=_spec(Web4EvaluationDecision.REJECT),
        owner_private_key=owner,
    )

    assert decision.research_evaluation_evidence_accepted is False
    assert decision.research_comparison_eligible is False
    assert decision.model_execution_authorized is False
    assert decision.training_authorization is False
    assert decision.promotion_eligible is False
    assert decision.production_activation_allowed is False


def test_owner_can_hold_passing_evidence_without_accepting_it() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner)
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=_spec(Web4EvaluationDecision.HOLD),
        owner_private_key=owner,
    )

    assert decision.evaluation_passed is True
    assert decision.research_evaluation_evidence_accepted is False
    assert decision.research_comparison_eligible is False


def test_release_owner_key_is_required() -> None:
    owner = Ed25519PrivateKey.generate()
    wrong_owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner)

    with pytest.raises(ValueError, match="owner key differs"):
        sign_web4_holdout_evaluation_decision(
            evidence=evidence,
            spec=_spec(Web4EvaluationDecision.ACCEPT),
            owner_private_key=wrong_owner,
        )


def test_decision_cannot_be_replayed_against_different_evidence() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner, identity_suffix="a")
    other_evidence = _evidence(owner, identity_suffix="2")
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=_spec(Web4EvaluationDecision.ACCEPT),
        owner_private_key=owner,
    )

    with pytest.raises(ValueError, match="binding differs"):
        verify_web4_holdout_evaluation_decision(
            evidence=other_evidence,
            decision=decision,
            owner_public_key=owner.public_key(),
        )


def test_rehashed_decision_tamper_fails_cryptographic_signature() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner)
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=_spec(Web4EvaluationDecision.ACCEPT),
        owner_private_key=owner,
    )
    tampered = decision.model_dump(mode="json")
    tampered["rationale"] = "Attacker rewrote rationale and recomputed public hashes without the owner key."

    semantic = dict(tampered)
    semantic.pop("artifact_sha256")
    semantic.pop("decision_sha256")
    semantic.pop("signature_algorithm")
    semantic.pop("owner_signature_base64")
    semantic.pop("owner_signature_verified")
    tampered["decision_sha256"] = _digest(semantic)
    signed = dict(tampered)
    signed.pop("artifact_sha256")
    tampered["artifact_sha256"] = _digest(signed)
    parsed = Web4HoldoutEvaluationDecision.model_validate(tampered)

    with pytest.raises(ValueError, match="signature verification failed"):
        verify_web4_holdout_evaluation_decision(
            evidence=evidence,
            decision=parsed,
            owner_public_key=owner.public_key(),
        )


def test_decision_self_hash_tampering_is_rejected() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner)
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=_spec(Web4EvaluationDecision.ACCEPT),
        owner_private_key=owner,
    )
    tampered = decision.model_dump(mode="json")
    tampered["approver_id"] = "attacker"

    with pytest.raises(ValueError):
        Web4HoldoutEvaluationDecision.model_validate(tampered)


def test_signed_decision_contains_no_answer_key_values() -> None:
    owner = Ed25519PrivateKey.generate()
    evidence = _evidence(owner)
    decision = sign_web4_holdout_evaluation_decision(
        evidence=evidence,
        spec=_spec(Web4EvaluationDecision.ACCEPT),
        owner_private_key=owner,
    )
    serialized = json.dumps(decision.model_dump(mode="json"), sort_keys=True)

    assert "REJECT_UNAUTHORIZED_WIDENING" not in serialized
    assert "AUTHORITATIVE_GUIDANCE_DRAFT_NOT_PROTOCOL_STANDARD" not in serialized
    assert base64.b64decode(decision.owner_signature_base64, validate=True)
