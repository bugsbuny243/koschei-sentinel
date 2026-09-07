from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkSplit,
    build_web4_benchmark_intake,
)
from koschei_sentinel.web4_benchmark_review import (
    Web4AdjudicationDecision,
    Web4BenchmarkAdjudication,
    Web4BenchmarkAdjudicationSpec,
    Web4BenchmarkHumanReview,
    Web4BenchmarkHumanReviewSpec,
    Web4HumanReviewDecision,
    verify_web4_benchmark_adjudication,
    verify_web4_benchmark_human_review,
)
from koschei_sentinel.web4_review_ops_cli import main as review_ops_main
from koschei_sentinel.web4_reviewer_trust import (
    Web4ReviewRole,
    load_web4_reviewer_trust_policy,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/web4/benchmark-intake"
_PROPOSAL = _FIXTURE / "proposal.json"
_ANSWER_KEY = _FIXTURE / "answer-key.json"
_SOURCES = _ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl"
_BENCHMARK = _ROOT / "evals/web4-security-benchmark.v1.json"
_INTAKE_POLICY = _ROOT / "evals/web4-benchmark-intake-policy.v1.json"


def _write_private_key(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _write_public_key(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _build_holdout_packet(tmp_path: Path):
    proposal_template = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    answer_template = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    proposal_path = tmp_path / "proposal.json"
    answer_path = tmp_path / "answer-key.json"
    for index in range(1, 200):
        case_id = f"fixture-web4-ops-{index:03d}"
        proposal = dict(proposal_template)
        proposal["case_id"] = case_id
        answer = dict(answer_template)
        answer["case_id"] = case_id
        proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
        answer_path.write_text(json.dumps(answer), encoding="utf-8")
        packet = build_web4_benchmark_intake(
            proposal_path=proposal_path,
            answer_key_path=answer_path,
            source_registry_path=_SOURCES,
            benchmark_policy_path=_BENCHMARK,
            intake_policy_path=_INTAKE_POLICY,
            snapshot_root=_FIXTURE,
        )
        if packet.split is Web4BenchmarkSplit.HOLDOUT:
            packet_path = tmp_path / "packet.json"
            packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
            return packet, packet_path, answer_path
    raise AssertionError("fixture search did not find deterministic HOLDOUT case")


def _operator_fixture(tmp_path: Path):
    packet, packet_path, answer_path = _build_holdout_packet(tmp_path)
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    adjudicator = Ed25519PrivateKey.generate()

    owner_private = tmp_path / "owner-private.pem"
    owner_public = tmp_path / "owner-public.pem"
    reviewer_private = tmp_path / "reviewer-private.pem"
    reviewer_public = tmp_path / "reviewer-public.pem"
    adjudicator_private = tmp_path / "adjudicator-private.pem"
    adjudicator_public = tmp_path / "adjudicator-public.pem"
    _write_private_key(owner_private, owner)
    _write_public_key(owner_public, owner)
    _write_private_key(reviewer_private, reviewer)
    _write_public_key(reviewer_public, reviewer)
    _write_private_key(adjudicator_private, adjudicator)
    _write_public_key(adjudicator_public, adjudicator)

    review_spec = tmp_path / "review-spec.json"
    review_spec.write_text(
        Web4BenchmarkHumanReviewSpec(
            reviewer_id="ops-reviewer",
            decision=Web4HumanReviewDecision.APPROVE,
            rationale="Operator primary review verified source, answer, authority, and family bindings.",
            source_refs_verified=True,
            source_revision_status_verified=True,
            answer_key_verified=True,
            authority_status_checked=True,
            benchmark_family_checked=True,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    adjudication_spec = tmp_path / "adjudication-spec.json"
    adjudication_spec.write_text(
        Web4BenchmarkAdjudicationSpec(
            adjudicator_id="ops-adjudicator",
            decision=Web4AdjudicationDecision.CONFIRM_PRIMARY,
            rationale="Independent operator adjudication rechecked answer, source bindings, and reasoning.",
            independently_checked_answer_key=True,
            independently_checked_source_bindings=True,
            independently_checked_review_reasoning=True,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return {
        "packet": packet,
        "packet_path": packet_path,
        "answer_path": answer_path,
        "owner": owner,
        "owner_private": owner_private,
        "owner_public": owner_public,
        "reviewer": reviewer,
        "reviewer_private": reviewer_private,
        "reviewer_public": reviewer_public,
        "adjudicator": adjudicator,
        "adjudicator_private": adjudicator_private,
        "adjudicator_public": adjudicator_public,
        "review_spec": review_spec,
        "adjudication_spec": adjudication_spec,
    }


def test_operator_cli_builds_trust_review_and_independent_adjudication(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _operator_fixture(tmp_path)
    reviewer_policy_path = tmp_path / "reviewer-trust.json"
    adjudicator_policy_path = tmp_path / "adjudicator-trust.json"

    assert review_ops_main(
        [
            "trust",
            "--delegate-public-key",
            str(fixture["reviewer_public"]),
            "--owner-private-key",
            str(fixture["owner_private"]),
            "--policy-id",
            "web4-ops-reviewer-v1",
            "--role",
            Web4ReviewRole.PRIMARY_REVIEWER.value,
            "--delegate-id",
            "ops-reviewer",
            "--output",
            str(reviewer_policy_path),
        ]
    ) == 0
    assert review_ops_main(
        [
            "trust",
            "--delegate-public-key",
            str(fixture["adjudicator_public"]),
            "--owner-private-key",
            str(fixture["owner_private"]),
            "--policy-id",
            "web4-ops-adjudicator-v1",
            "--role",
            Web4ReviewRole.ADJUDICATOR.value,
            "--delegate-id",
            "ops-adjudicator",
            "--output",
            str(adjudicator_policy_path),
        ]
    ) == 0

    review_path = tmp_path / "review.json"
    assert review_ops_main(
        [
            "review",
            "--packet",
            str(fixture["packet_path"]),
            "--answer-key",
            str(fixture["answer_path"]),
            "--spec",
            str(fixture["review_spec"]),
            "--reviewer-private-key",
            str(fixture["reviewer_private"]),
            "--reviewer-trust-policy",
            str(reviewer_policy_path),
            "--owner-public-key",
            str(fixture["owner_public"]),
            "--output",
            str(review_path),
        ]
    ) == 0
    review = Web4BenchmarkHumanReview.model_validate_json(review_path.read_bytes())
    reviewer_policy = load_web4_reviewer_trust_policy(reviewer_policy_path)
    verify_web4_benchmark_human_review(
        review=review,
        packet=fixture["packet"],
        answer_key_path=fixture["answer_path"],
        reviewer_public_key=fixture["reviewer"].public_key(),
        trust_policy=reviewer_policy,
        owner_public_key=fixture["owner"].public_key(),
    )

    adjudication_path = tmp_path / "adjudication.json"
    assert review_ops_main(
        [
            "adjudicate",
            "--packet",
            str(fixture["packet_path"]),
            "--answer-key",
            str(fixture["answer_path"]),
            "--review",
            str(review_path),
            "--reviewer-public-key",
            str(fixture["reviewer_public"]),
            "--reviewer-trust-policy",
            str(reviewer_policy_path),
            "--spec",
            str(fixture["adjudication_spec"]),
            "--adjudicator-private-key",
            str(fixture["adjudicator_private"]),
            "--adjudicator-trust-policy",
            str(adjudicator_policy_path),
            "--owner-public-key",
            str(fixture["owner_public"]),
            "--output",
            str(adjudication_path),
        ]
    ) == 0
    adjudication = Web4BenchmarkAdjudication.model_validate_json(adjudication_path.read_bytes())
    adjudicator_policy = load_web4_reviewer_trust_policy(adjudicator_policy_path)
    verify_web4_benchmark_adjudication(
        adjudication=adjudication,
        review=review,
        packet=fixture["packet"],
        answer_key_path=fixture["answer_path"],
        reviewer_public_key=fixture["reviewer"].public_key(),
        reviewer_trust_policy=reviewer_policy,
        adjudicator_public_key=fixture["adjudicator"].public_key(),
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=fixture["owner"].public_key(),
    )

    assert adjudication.independent_adjudication is True
    assert adjudication.final_review_approved is True
    assert adjudication.eligible_for_signed_holdout_release is True
    assert adjudication.training_authorization is False
    assert adjudication.evaluation_authorization is False
    assert adjudication.promotion_eligible is False

    output = capsys.readouterr()
    combined = output.out + output.err
    assert "REJECT_UNAUTHORIZED_WIDENING" not in combined
    assert "AUTHORITATIVE_GUIDANCE_DRAFT_NOT_PROTOCOL_STANDARD" not in combined
    assert "PRIVATE KEY" not in combined


def test_operator_cli_refuses_existing_output_before_key_reads(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve-me", encoding="utf-8")

    assert review_ops_main(
        [
            "trust",
            "--delegate-public-key",
            str(tmp_path / "missing-public.pem"),
            "--owner-private-key",
            str(tmp_path / "missing-owner.pem"),
            "--policy-id",
            "web4-ops-existing-v1",
            "--role",
            Web4ReviewRole.PRIMARY_REVIEWER.value,
            "--delegate-id",
            "ops-reviewer",
            "--output",
            str(output),
        ]
    ) == 2
    assert output.read_text(encoding="utf-8") == "preserve-me"


def test_operator_cli_rejects_symlink_owner_private_key(tmp_path: Path) -> None:
    owner = Ed25519PrivateKey.generate()
    delegate = Ed25519PrivateKey.generate()
    owner_private = tmp_path / "owner.pem"
    owner_link = tmp_path / "owner-link.pem"
    delegate_public = tmp_path / "delegate.pem"
    _write_private_key(owner_private, owner)
    _write_public_key(delegate_public, delegate)
    owner_link.symlink_to(owner_private)
    output = tmp_path / "trust.json"

    assert review_ops_main(
        [
            "trust",
            "--delegate-public-key",
            str(delegate_public),
            "--owner-private-key",
            str(owner_link),
            "--policy-id",
            "web4-ops-symlink-v1",
            "--role",
            Web4ReviewRole.PRIMARY_REVIEWER.value,
            "--delegate-id",
            "ops-reviewer",
            "--output",
            str(output),
        ]
    ) == 2
    assert not output.exists()


def test_operator_cli_rejects_symlink_reviewer_private_key(tmp_path: Path) -> None:
    fixture = _operator_fixture(tmp_path)
    reviewer_policy_path = tmp_path / "reviewer-trust.json"
    assert review_ops_main(
        [
            "trust",
            "--delegate-public-key",
            str(fixture["reviewer_public"]),
            "--owner-private-key",
            str(fixture["owner_private"]),
            "--policy-id",
            "web4-ops-reviewer-v1",
            "--role",
            Web4ReviewRole.PRIMARY_REVIEWER.value,
            "--delegate-id",
            "ops-reviewer",
            "--output",
            str(reviewer_policy_path),
        ]
    ) == 0
    reviewer_link = tmp_path / "reviewer-link.pem"
    reviewer_link.symlink_to(fixture["reviewer_private"])
    review_path = tmp_path / "review.json"

    assert review_ops_main(
        [
            "review",
            "--packet",
            str(fixture["packet_path"]),
            "--answer-key",
            str(fixture["answer_path"]),
            "--spec",
            str(fixture["review_spec"]),
            "--reviewer-private-key",
            str(reviewer_link),
            "--reviewer-trust-policy",
            str(reviewer_policy_path),
            "--owner-public-key",
            str(fixture["owner_public"]),
            "--output",
            str(review_path),
        ]
    ) == 2
    assert not review_path.exists()


def test_operator_cli_does_not_overwrite_review_artifact(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text("immutable-review", encoding="utf-8")

    assert review_ops_main(
        [
            "review",
            "--packet",
            str(tmp_path / "missing-packet.json"),
            "--answer-key",
            str(tmp_path / "missing-answer.json"),
            "--spec",
            str(tmp_path / "missing-spec.json"),
            "--reviewer-private-key",
            str(tmp_path / "missing-private.pem"),
            "--reviewer-trust-policy",
            str(tmp_path / "missing-policy.json"),
            "--owner-public-key",
            str(tmp_path / "missing-owner.pem"),
            "--output",
            str(review_path),
        ]
    ) == 2
    assert review_path.read_text(encoding="utf-8") == "immutable-review"
