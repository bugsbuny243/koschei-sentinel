from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkAnswerKey,
    Web4BenchmarkSplit,
    build_web4_benchmark_intake,
)
from koschei_sentinel.web4_benchmark_review import (
    Web4AdjudicationDecision,
    Web4BenchmarkAdjudicationSpec,
    Web4BenchmarkHumanReviewSpec,
    Web4HumanReviewDecision,
    adjudicate_web4_benchmark_review,
    sign_web4_benchmark_human_review,
)
from koschei_sentinel.web4_holdout_evaluation import (
    Web4HoldoutEvaluationEvidence,
    Web4HoldoutInferenceCase,
    Web4HoldoutPrediction,
    build_web4_holdout_evaluation_evidence,
    build_web4_holdout_inference_pack,
    build_web4_holdout_prediction,
    build_web4_holdout_prediction_set,
    evaluate_web4_holdout_predictions,
)
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutReleaseMaterial,
    build_web4_holdout_release,
)
from koschei_sentinel.web4_reviewer_trust import (
    Web4ReviewRole,
    build_web4_reviewer_trust_policy,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/web4/benchmark-intake"
_PROPOSAL = _FIXTURE / "proposal.json"
_ANSWER_KEY = _FIXTURE / "answer-key.json"
_SOURCES = _ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl"
_BENCHMARK = _ROOT / "evals/web4-security-benchmark.v1.json"
_INTAKE_POLICY = _ROOT / "evals/web4-benchmark-intake-policy.v1.json"
_MODEL_ARTIFACT_SHA = hashlib.sha256(b"web4-offline-replay-fixture-model").hexdigest()


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _build_signed_release(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    proposal_template = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    answer_template = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    packet = None
    answer_path = None
    for index in range(1, 200):
        case_id = f"fixture-web4-eval-{index:03d}"
        proposal = dict(proposal_template)
        proposal["case_id"] = case_id
        answer = dict(answer_template)
        answer["case_id"] = case_id
        proposal_path = tmp_path / "proposal.json"
        candidate_answer_path = tmp_path / "answer-key.json"
        proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
        candidate_answer_path.write_text(json.dumps(answer), encoding="utf-8")
        candidate = build_web4_benchmark_intake(
            proposal_path=proposal_path,
            answer_key_path=candidate_answer_path,
            source_registry_path=_SOURCES,
            benchmark_policy_path=_BENCHMARK,
            intake_policy_path=_INTAKE_POLICY,
            snapshot_root=_FIXTURE,
        )
        if candidate.split is Web4BenchmarkSplit.HOLDOUT:
            packet = candidate
            answer_path = candidate_answer_path
            break
    if packet is None or answer_path is None:
        raise AssertionError("fixture search did not find deterministic HOLDOUT case")

    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    adjudicator = Ed25519PrivateKey.generate()
    reviewer_policy = build_web4_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="web4-eval-reviewer-v1",
        role=Web4ReviewRole.PRIMARY_REVIEWER,
        delegate_id="eval-reviewer",
    )
    adjudicator_policy = build_web4_reviewer_trust_policy(
        adjudicator.public_key(),
        owner,
        policy_id="web4-eval-adjudicator-v1",
        role=Web4ReviewRole.ADJUDICATOR,
        delegate_id="eval-adjudicator",
    )
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=answer_path,
        spec=Web4BenchmarkHumanReviewSpec(
            reviewer_id="eval-reviewer",
            decision=Web4HumanReviewDecision.APPROVE,
            rationale="Primary review verified the benchmark answer, provenance, and authority boundary.",
            source_refs_verified=True,
            source_revision_status_verified=True,
            answer_key_verified=True,
            authority_status_checked=True,
            benchmark_family_checked=True,
        ),
        reviewer_private_key=reviewer,
        trust_policy=reviewer_policy,
        owner_public_key=owner.public_key(),
    )
    adjudication = adjudicate_web4_benchmark_review(
        review=review,
        packet=packet,
        answer_key_path=answer_path,
        reviewer_public_key=reviewer.public_key(),
        reviewer_trust_policy=reviewer_policy,
        adjudication_spec=Web4BenchmarkAdjudicationSpec(
            adjudicator_id="eval-adjudicator",
            decision=Web4AdjudicationDecision.CONFIRM_PRIMARY,
            rationale="Independent adjudication confirmed the answer key and signed source bindings.",
            independently_checked_answer_key=True,
            independently_checked_source_bindings=True,
            independently_checked_review_reasoning=True,
        ),
        adjudicator_private_key=adjudicator,
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=owner.public_key(),
    )
    material = Web4HoldoutReleaseMaterial(
        packet=packet,
        review=review,
        adjudication=adjudication,
        answer_key_path=answer_path,
        reviewer_public_key=reviewer.public_key(),
        reviewer_trust_policy=reviewer_policy,
        adjudicator_public_key=adjudicator.public_key(),
        adjudicator_trust_policy=adjudicator_policy,
    )
    release = build_web4_holdout_release(
        release_id="web4-eval-fixture-v1",
        materials=[material],
        benchmark_policy_path=_BENCHMARK,
        owner_private_key=owner,
    )
    answer_dir = tmp_path / "answer-keys"
    answer_dir.mkdir()
    isolated_answer = answer_dir / f"{packet.case_id}.json"
    isolated_answer.write_bytes(answer_path.read_bytes())
    return owner, release, answer_dir


def _perfect_prediction_stack(tmp_path: Path):
    owner, release, answer_dir = _build_signed_release(tmp_path)
    pack = build_web4_holdout_inference_pack(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )
    answer_key = Web4BenchmarkAnswerKey.model_validate_json(
        next(answer_dir.iterdir()).read_bytes()
    )
    prediction = build_web4_holdout_prediction(
        inference_case=pack.cases[0],
        release_sha256=release.release_sha256,
        model_ref="fixture/web4-offline-replay",
        model_revision="fixture-v1",
        model_artifact_sha256=_MODEL_ARTIFACT_SHA,
        answer=answer_key.expected,
    )
    prediction_set = build_web4_holdout_prediction_set(
        inference_pack=pack,
        predictions=[prediction],
        model_ref="fixture/web4-offline-replay",
        model_revision="fixture-v1",
        model_artifact_sha256=_MODEL_ARTIFACT_SHA,
    )
    return owner, release, answer_dir, pack, prediction_set


def test_inference_pack_excludes_answer_key_values(tmp_path: Path) -> None:
    _, release, answer_dir = _build_signed_release(tmp_path)
    owner = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError):
        build_web4_holdout_inference_pack(
            release=release,
            owner_public_key=owner.public_key(),
            benchmark_policy_path=_BENCHMARK,
        )

    owner, release, answer_dir = _build_signed_release(tmp_path / "valid")
    pack = build_web4_holdout_inference_pack(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )
    serialized = json.dumps(pack.model_dump(mode="json"), sort_keys=True)
    assert pack.answer_key_excluded is True
    assert pack.network_access_required is False
    assert pack.gpu_required is False
    assert "REJECT_UNAUTHORIZED_WIDENING" not in serialized
    assert "AUTHORITATIVE_GUIDANCE_DRAFT_NOT_PROTOCOL_STANDARD" not in serialized


def test_perfect_offline_replay_builds_passing_nonleaking_evidence(tmp_path: Path) -> None:
    owner, release, answer_dir, pack, prediction_set = _perfect_prediction_stack(tmp_path)
    evidence = build_web4_holdout_evaluation_evidence(
        release=release,
        inference_pack=pack,
        prediction_set=prediction_set,
        answer_key_dir=answer_dir,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )

    assert evidence.passed is True
    assert evidence.complete_case_accounting is True
    assert evidence.report.exact_match_rate == 1.0
    assert evidence.report.expected_field_accuracy == 1.0
    assert evidence.answer_key_values_embedded is False
    assert evidence.offline_replay is True
    assert evidence.network_access_required is False
    assert evidence.gpu_required is False
    assert evidence.training_authorization is False
    assert evidence.promotion_eligible is False
    assert evidence.production_activation_allowed is False
    serialized = json.dumps(evidence.model_dump(mode="json"), sort_keys=True)
    assert "REJECT_UNAUTHORIZED_WIDENING" not in serialized
    assert "AUTHORITATIVE_GUIDANCE_DRAFT_NOT_PROTOCOL_STANDARD" not in serialized


def test_wrong_prediction_fails_without_disclosing_expected_values(tmp_path: Path) -> None:
    owner, release, answer_dir = _build_signed_release(tmp_path)
    pack = build_web4_holdout_inference_pack(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )
    prediction = build_web4_holdout_prediction(
        inference_case=pack.cases[0],
        release_sha256=release.release_sha256,
        model_ref="fixture/web4-offline-replay",
        model_revision="fixture-v1",
        model_artifact_sha256=_MODEL_ARTIFACT_SHA,
        answer={
            "delegation_scope_result": "ALLOW_WIDENING",
            "authority_status": "STANDARD",
            "must_request_new_grant_for_wider_scope": False,
        },
    )
    prediction_set = build_web4_holdout_prediction_set(
        inference_pack=pack,
        predictions=[prediction],
        model_ref="fixture/web4-offline-replay",
        model_revision="fixture-v1",
        model_artifact_sha256=_MODEL_ARTIFACT_SHA,
    )
    report, _ = evaluate_web4_holdout_predictions(
        release=release,
        inference_pack=pack,
        prediction_set=prediction_set,
        answer_key_dir=answer_dir,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )

    assert report.passed is False
    assert report.exact_match_rate == 0.0
    assert report.expected_field_accuracy == 0.0
    assert report.case_results[0].mismatched_field_paths
    serialized = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    assert "REJECT_UNAUTHORIZED_WIDENING" not in serialized
    assert "AUTHORITATIVE_GUIDANCE_DRAFT_NOT_PROTOCOL_STANDARD" not in serialized


def test_missing_prediction_fails_complete_case_accounting(tmp_path: Path) -> None:
    owner, release, answer_dir = _build_signed_release(tmp_path)
    pack = build_web4_holdout_inference_pack(
        release=release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )
    prediction_set = build_web4_holdout_prediction_set(
        inference_pack=pack,
        predictions=[],
        model_ref="fixture/web4-offline-replay",
        model_revision="fixture-v1",
        model_artifact_sha256=_MODEL_ARTIFACT_SHA,
    )
    report, _ = evaluate_web4_holdout_predictions(
        release=release,
        inference_pack=pack,
        prediction_set=prediction_set,
        answer_key_dir=answer_dir,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )

    assert report.passed is False
    assert report.complete_case_accounting is False
    assert report.missing_case_ids == release.case_ids


def test_extra_prediction_fails_complete_case_accounting(tmp_path: Path) -> None:
    owner, release, answer_dir, pack, prediction_set = _perfect_prediction_stack(tmp_path)
    fake_case = Web4HoldoutInferenceCase(
        case_id="unexpected-case",
        family=pack.cases[0].family,
        release_case_sha256="f" * 64,
        model_input={"messages": [{"role": "user", "content": "untrusted extra"}]},
        model_input_sha256="e" * 64,
    )
    extra = build_web4_holdout_prediction(
        inference_case=fake_case,
        release_sha256=release.release_sha256,
        model_ref="fixture/web4-offline-replay",
        model_revision="fixture-v1",
        model_artifact_sha256=_MODEL_ARTIFACT_SHA,
        answer={"unexpected": True},
    )
    prediction_set = build_web4_holdout_prediction_set(
        inference_pack=pack,
        predictions=[*prediction_set.predictions, extra],
        model_ref="fixture/web4-offline-replay",
        model_revision="fixture-v1",
        model_artifact_sha256=_MODEL_ARTIFACT_SHA,
    )
    report, _ = evaluate_web4_holdout_predictions(
        release=release,
        inference_pack=pack,
        prediction_set=prediction_set,
        answer_key_dir=answer_dir,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )

    assert report.passed is False
    assert report.extra_case_ids == ["unexpected-case"]
    assert report.complete_case_accounting is False


def test_answer_key_drift_is_rejected(tmp_path: Path) -> None:
    owner, release, answer_dir, pack, prediction_set = _perfect_prediction_stack(tmp_path)
    answer_path = next(answer_dir.iterdir())
    payload = json.loads(answer_path.read_text(encoding="utf-8"))
    payload["expected"]["authority_status"] = "STANDARD"
    answer_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="answer key drifted"):
        evaluate_web4_holdout_predictions(
            release=release,
            inference_pack=pack,
            prediction_set=prediction_set,
            answer_key_dir=answer_dir,
            owner_public_key=owner.public_key(),
            benchmark_policy_path=_BENCHMARK,
        )


def test_prediction_self_hash_tampering_is_rejected(tmp_path: Path) -> None:
    _, _, _, _, prediction_set = _perfect_prediction_stack(tmp_path)
    tampered = prediction_set.predictions[0].model_dump(mode="json")
    tampered["answer"]["authority_status"] = "STANDARD"

    with pytest.raises(ValueError, match="prediction self-hash"):
        Web4HoldoutPrediction.model_validate(tampered)


def test_mixed_model_identity_is_rejected(tmp_path: Path) -> None:
    owner, release, _, pack, prediction_set = _perfect_prediction_stack(tmp_path)
    original = prediction_set.predictions[0]
    different = build_web4_holdout_prediction(
        inference_case=pack.cases[0],
        release_sha256=release.release_sha256,
        model_ref="fixture/different-model",
        model_revision="other-v1",
        model_artifact_sha256="d" * 64,
        answer=original.answer,
    )

    with pytest.raises(ValueError, match="model identity differs"):
        build_web4_holdout_prediction_set(
            inference_pack=pack,
            predictions=[different],
            model_ref="fixture/web4-offline-replay",
            model_revision="fixture-v1",
            model_artifact_sha256=_MODEL_ARTIFACT_SHA,
        )


def test_evidence_self_hash_tampering_is_rejected(tmp_path: Path) -> None:
    owner, release, answer_dir, pack, prediction_set = _perfect_prediction_stack(tmp_path)
    evidence = build_web4_holdout_evaluation_evidence(
        release=release,
        inference_pack=pack,
        prediction_set=prediction_set,
        answer_key_dir=answer_dir,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
    )
    tampered = evidence.model_dump(mode="json")
    tampered["model_revision"] = "tampered"

    with pytest.raises(ValueError):
        Web4HoldoutEvaluationEvidence.model_validate(tampered)
