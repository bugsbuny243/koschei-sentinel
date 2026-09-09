from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
from koschei_sentinel.web4_holdout_evaluation import Web4HoldoutEvaluationEvidence
from koschei_sentinel.web4_holdout_evaluation_decision import (
    Web4EvaluationDecision,
    Web4HoldoutEvaluationDecision,
    verify_web4_holdout_evaluation_decision,
)
from koschei_sentinel.web4_holdout_ops_cli import main
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutRelease,
    verify_web4_holdout_release,
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
_MODEL_SHA = hashlib.sha256(b"web4-holdout-ops-offline-model").hexdigest()


def _write_private(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_public(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _write_model(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_material_bundle(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    proposal_template = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    answer_template = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    packet = None
    answer_path = None
    for index in range(1, 300):
        case_id = f"fixture-web4-ops-{index:03d}"
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
        policy_id="web4-ops-reviewer-v1",
        role=Web4ReviewRole.PRIMARY_REVIEWER,
        delegate_id="ops-reviewer",
    )
    adjudicator_policy = build_web4_reviewer_trust_policy(
        adjudicator.public_key(),
        owner,
        policy_id="web4-ops-adjudicator-v1",
        role=Web4ReviewRole.ADJUDICATOR,
        delegate_id="ops-adjudicator",
    )
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=answer_path,
        spec=Web4BenchmarkHumanReviewSpec(
            reviewer_id="ops-reviewer",
            decision=Web4HumanReviewDecision.APPROVE,
            rationale="Primary operator review verified provenance, answer key, and authority status.",
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
            adjudicator_id="ops-adjudicator",
            decision=Web4AdjudicationDecision.CONFIRM_PRIMARY,
            rationale="Independent operator adjudication rechecked answer and provenance bindings.",
            independently_checked_answer_key=True,
            independently_checked_source_bindings=True,
            independently_checked_review_reasoning=True,
        ),
        adjudicator_private_key=adjudicator,
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=owner.public_key(),
    )

    owner_private = tmp_path / "owner-private.pem"
    owner_public = tmp_path / "owner-public.pem"
    reviewer_public = tmp_path / "reviewer-public.pem"
    adjudicator_public = tmp_path / "adjudicator-public.pem"
    _write_private(owner_private, owner)
    _write_public(owner_public, owner)
    _write_public(reviewer_public, reviewer)
    _write_public(adjudicator_public, adjudicator)

    packet_path = tmp_path / "packet.json"
    review_path = tmp_path / "review.json"
    adjudication_path = tmp_path / "adjudication.json"
    reviewer_policy_path = tmp_path / "reviewer-policy.json"
    adjudicator_policy_path = tmp_path / "adjudicator-policy.json"
    _write_model(packet_path, packet)
    _write_model(review_path, review)
    _write_model(adjudication_path, adjudication)
    _write_model(reviewer_policy_path, reviewer_policy)
    _write_model(adjudicator_policy_path, adjudicator_policy)

    manifest = {
        "schema_version": "sentinel.web4-holdout-release-material-manifest.v1",
        "cases": [
            {
                "packet": packet_path.name,
                "review": review_path.name,
                "adjudication": adjudication_path.name,
                "answer_key": answer_path.name,
                "reviewer_public_key": reviewer_public.name,
                "reviewer_trust_policy": reviewer_policy_path.name,
                "adjudicator_public_key": adjudicator_public.name,
                "adjudicator_trust_policy": adjudicator_policy_path.name,
            }
        ],
    }
    manifest_path = tmp_path / "material-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    answer_dir = tmp_path / "answer-keys"
    answer_dir.mkdir()
    isolated_answer = answer_dir / f"{packet.case_id}.json"
    isolated_answer.write_bytes(answer_path.read_bytes())
    return {
        "owner": owner,
        "owner_private": owner_private,
        "owner_public": owner_public,
        "manifest": manifest_path,
        "packet": packet,
        "answer_dir": answer_dir,
        "answer_path": isolated_answer,
    }


def test_full_offline_operator_chain_is_signed_nonleaking_and_nonactivating(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = _build_material_bundle(tmp_path)
    release_path = tmp_path / "release.json"
    pack_path = tmp_path / "pack.json"
    predictions_path = tmp_path / "predictions.json"
    evidence_path = tmp_path / "evidence.json"
    decision_path = tmp_path / "decision.json"

    assert main(
        [
            "release",
            "--release-id",
            "web4-ops-release-v1",
            "--material-manifest",
            str(bundle["manifest"]),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-private-key",
            str(bundle["owner_private"]),
            "--output",
            str(release_path),
        ]
    ) == 0
    assert main(
        [
            "pack",
            "--release",
            str(release_path),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-public-key",
            str(bundle["owner_public"]),
            "--output",
            str(pack_path),
        ]
    ) == 0

    answer_key = Web4BenchmarkAnswerKey.model_validate_json(
        Path(bundle["answer_path"]).read_bytes()
    )
    prediction_manifest = {
        "schema_version": "sentinel.web4-offline-prediction-manifest.v1",
        "model_ref": "fixture/web4-offline-ops",
        "model_revision": "fixture-v1",
        "model_artifact_sha256": _MODEL_SHA,
        "predictions": [
            {"case_id": bundle["packet"].case_id, "answer": answer_key.expected}
        ],
    }
    prediction_input = tmp_path / "prediction-input.json"
    prediction_input.write_text(json.dumps(prediction_manifest), encoding="utf-8")
    assert main(
        [
            "predictions",
            "--inference-pack",
            str(pack_path),
            "--input",
            str(prediction_input),
            "--output",
            str(predictions_path),
        ]
    ) == 0
    assert main(
        [
            "evaluate",
            "--release",
            str(release_path),
            "--inference-pack",
            str(pack_path),
            "--prediction-set",
            str(predictions_path),
            "--answer-key-dir",
            str(bundle["answer_dir"]),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-public-key",
            str(bundle["owner_public"]),
            "--output",
            str(evidence_path),
        ]
    ) == 0

    decision_spec = {
        "schema_version": "sentinel.web4-holdout-evaluation-decision-spec.v1",
        "decision_id": "web4-ops-decision-v1",
        "approver_id": "owner-operator",
        "decision": Web4EvaluationDecision.ACCEPT.value,
        "rationale": "Passing complete research evidence is accepted for comparison only.",
    }
    decision_spec_path = tmp_path / "decision-spec.json"
    decision_spec_path.write_text(json.dumps(decision_spec), encoding="utf-8")
    assert main(
        [
            "decide",
            "--evidence",
            str(evidence_path),
            "--spec",
            str(decision_spec_path),
            "--owner-private-key",
            str(bundle["owner_private"]),
            "--output",
            str(decision_path),
        ]
    ) == 0

    release = Web4HoldoutRelease.model_validate_json(release_path.read_bytes())
    verified_release = verify_web4_holdout_release(
        release,
        owner_public_key=bundle["owner"].public_key(),
        benchmark_policy_path=_BENCHMARK,
    )
    evidence = Web4HoldoutEvaluationEvidence.model_validate_json(evidence_path.read_bytes())
    decision = Web4HoldoutEvaluationDecision.model_validate_json(decision_path.read_bytes())
    verified_decision = verify_web4_holdout_evaluation_decision(
        evidence=evidence,
        decision=decision,
        owner_public_key=bundle["owner"].public_key(),
    )

    assert verified_release.training_authorization is False
    assert evidence.passed is True
    assert evidence.answer_key_values_embedded is False
    assert evidence.network_access_required is False
    assert evidence.gpu_required is False
    assert evidence.training_authorization is False
    assert evidence.promotion_eligible is False
    assert verified_decision.research_comparison_eligible is True
    assert verified_decision.model_execution_authorized is False
    assert verified_decision.training_authorization is False
    assert verified_decision.promotion_eligible is False
    assert verified_decision.production_activation_allowed is False

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "REJECT_UNAUTHORIZED_WIDENING" not in combined
    assert "AUTHORITATIVE_GUIDANCE_DRAFT_NOT_PROTOCOL_STANDARD" not in combined
    assert "BEGIN PRIVATE KEY" not in combined

    evidence_serialized = evidence_path.read_text(encoding="utf-8")
    assert "REJECT_UNAUTHORIZED_WIDENING" not in evidence_serialized
    assert "AUTHORITATIVE_GUIDANCE_DRAFT_NOT_PROTOCOL_STANDARD" not in evidence_serialized


def test_release_output_preflight_happens_before_private_key_read(tmp_path: Path) -> None:
    output = tmp_path / "release.json"
    output.write_text("sentinel", encoding="utf-8")
    result = main(
        [
            "release",
            "--release-id",
            "web4-preflight-v1",
            "--material-manifest",
            str(tmp_path / "missing-manifest.json"),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-private-key",
            str(tmp_path / "missing-owner.pem"),
            "--output",
            str(output),
        ]
    )
    assert result == 2
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_predictions_reject_unknown_case(tmp_path: Path) -> None:
    bundle = _build_material_bundle(tmp_path)
    release_path = tmp_path / "release.json"
    pack_path = tmp_path / "pack.json"
    assert main(
        [
            "release",
            "--release-id",
            "web4-unknown-case-v1",
            "--material-manifest",
            str(bundle["manifest"]),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-private-key",
            str(bundle["owner_private"]),
            "--output",
            str(release_path),
        ]
    ) == 0
    assert main(
        [
            "pack",
            "--release",
            str(release_path),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-public-key",
            str(bundle["owner_public"]),
            "--output",
            str(pack_path),
        ]
    ) == 0
    prediction_input = tmp_path / "prediction-input.json"
    prediction_input.write_text(
        json.dumps(
            {
                "schema_version": "sentinel.web4-offline-prediction-manifest.v1",
                "model_ref": "fixture/web4-offline-ops",
                "model_revision": "fixture-v1",
                "model_artifact_sha256": _MODEL_SHA,
                "predictions": [{"case_id": "not-in-pack", "answer": {"x": True}}],
            }
        ),
        encoding="utf-8",
    )
    assert main(
        [
            "predictions",
            "--inference-pack",
            str(pack_path),
            "--input",
            str(prediction_input),
            "--output",
            str(tmp_path / "predictions.json"),
        ]
    ) == 2


def test_release_rejects_symlink_owner_private_key(tmp_path: Path) -> None:
    bundle = _build_material_bundle(tmp_path)
    symlink = tmp_path / "owner-link.pem"
    symlink.symlink_to(Path(bundle["owner_private"]).name)
    assert main(
        [
            "release",
            "--release-id",
            "web4-symlink-owner-v1",
            "--material-manifest",
            str(bundle["manifest"]),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-private-key",
            str(symlink),
            "--output",
            str(tmp_path / "release.json"),
        ]
    ) == 2


def test_material_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "material-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "sentinel.web4-holdout-release-material-manifest.v1",
                "cases": [
                    {
                        "packet": "../packet.json",
                        "review": "review.json",
                        "adjudication": "adjudication.json",
                        "answer_key": "answer-key.json",
                        "reviewer_public_key": "reviewer.pem",
                        "reviewer_trust_policy": "reviewer-policy.json",
                        "adjudicator_public_key": "adjudicator.pem",
                        "adjudicator_trust_policy": "adjudicator-policy.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    owner = Ed25519PrivateKey.generate()
    owner_path = tmp_path / "owner.pem"
    _write_private(owner_path, owner)
    assert main(
        [
            "release",
            "--release-id",
            "web4-traversal-v1",
            "--material-manifest",
            str(manifest),
            "--benchmark-policy",
            str(_BENCHMARK),
            "--owner-private-key",
            str(owner_path),
            "--output",
            str(tmp_path / "release.json"),
        ]
    ) == 2
