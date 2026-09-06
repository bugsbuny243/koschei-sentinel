from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from koschei_sentinel.training import canonical_json
from koschei_sentinel.web4_benchmark_intake import (
    Web4BenchmarkIntakePacket,
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
    adjudicate_web4_benchmark_review,
    sign_web4_benchmark_human_review,
    verify_web4_benchmark_adjudication,
    verify_web4_benchmark_human_review,
)
from koschei_sentinel.web4_reviewer_trust import (
    Web4ReviewRole,
    build_web4_reviewer_trust_policy,
    verify_web4_reviewer_trust_policy,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "fixtures/web4/benchmark-intake"
_PROPOSAL = _FIXTURE / "proposal.json"
_ANSWER_KEY = _FIXTURE / "answer-key.json"
_SOURCES = _ROOT / "configs/corpus/web4-v1.sources.proposed.jsonl"
_BENCHMARK = _ROOT / "evals/web4-security-benchmark.v1.json"
_INTAKE_POLICY = _ROOT / "evals/web4-benchmark-intake-policy.v1.json"


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _build_packet(
    *,
    proposal_path: Path = _PROPOSAL,
    answer_key_path: Path = _ANSWER_KEY,
    snapshot_root: Path | None = None,
) -> Web4BenchmarkIntakePacket:
    return build_web4_benchmark_intake(
        proposal_path=proposal_path,
        answer_key_path=answer_key_path,
        source_registry_path=_SOURCES,
        benchmark_policy_path=_BENCHMARK,
        intake_policy_path=_INTAKE_POLICY,
        snapshot_root=snapshot_root,
    )


def _trust_pair():
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    adjudicator = Ed25519PrivateKey.generate()
    reviewer_policy = build_web4_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="web4-reviewer-fixture-v1",
        role=Web4ReviewRole.PRIMARY_REVIEWER,
        delegate_id="reviewer-fixture",
    )
    adjudicator_policy = build_web4_reviewer_trust_policy(
        adjudicator.public_key(),
        owner,
        policy_id="web4-adjudicator-fixture-v1",
        role=Web4ReviewRole.ADJUDICATOR,
        delegate_id="adjudicator-fixture",
    )
    return owner, reviewer, adjudicator, reviewer_policy, adjudicator_policy


def _approved_review_spec() -> Web4BenchmarkHumanReviewSpec:
    return Web4BenchmarkHumanReviewSpec(
        reviewer_id="reviewer-fixture",
        decision=Web4HumanReviewDecision.APPROVE,
        rationale=(
            "The case is source-bound, answer-key-consistent, and preserves the authority boundary."
        ),
        source_refs_verified=True,
        source_revision_status_verified=True,
        answer_key_verified=True,
        authority_status_checked=True,
        benchmark_family_checked=True,
    )


def _confirm_spec() -> Web4BenchmarkAdjudicationSpec:
    return Web4BenchmarkAdjudicationSpec(
        adjudicator_id="adjudicator-fixture",
        decision=Web4AdjudicationDecision.CONFIRM_PRIMARY,
        rationale="Independent review confirms the signed primary-review reasoning and bindings.",
        independently_checked_answer_key=True,
        independently_checked_source_bindings=True,
        independently_checked_review_reasoning=True,
    )


def _find_holdout_packet(tmp_path: Path) -> tuple[Web4BenchmarkIntakePacket, Path]:
    base_proposal = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    base_answer = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    for index in range(1, 200):
        case_id = f"fixture-web4-holdout-{index:03d}"
        proposal = dict(base_proposal)
        proposal["case_id"] = case_id
        answer = dict(base_answer)
        answer["case_id"] = case_id
        proposal_path = tmp_path / "proposal.json"
        answer_path = tmp_path / "answer-key.json"
        proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
        answer_path.write_text(json.dumps(answer), encoding="utf-8")
        packet = _build_packet(
            proposal_path=proposal_path,
            answer_key_path=answer_path,
            snapshot_root=_FIXTURE,
        )
        if packet.split is Web4BenchmarkSplit.HOLDOUT:
            return packet, answer_path
    raise AssertionError("fixture search did not find deterministic HOLDOUT case")


def test_owner_signed_role_trust_policies_verify() -> None:
    owner, reviewer, adjudicator, reviewer_policy, adjudicator_policy = _trust_pair()

    assert reviewer_policy.role is Web4ReviewRole.PRIMARY_REVIEWER
    assert adjudicator_policy.role is Web4ReviewRole.ADJUDICATOR
    verify_web4_reviewer_trust_policy(
        reviewer_policy,
        reviewer.public_key(),
        owner.public_key(),
    )
    verify_web4_reviewer_trust_policy(
        adjudicator_policy,
        adjudicator.public_key(),
        owner.public_key(),
    )
    assert reviewer_policy.delegate_key_fingerprint != adjudicator_policy.delegate_key_fingerprint


def test_trust_policy_rejects_wrong_owner_key() -> None:
    owner, reviewer, _, reviewer_policy, _ = _trust_pair()
    wrong_owner = Ed25519PrivateKey.generate()

    assert owner.public_key() != wrong_owner.public_key()
    with pytest.raises(ValueError, match="owner public key does not match"):
        verify_web4_reviewer_trust_policy(
            reviewer_policy,
            reviewer.public_key(),
            wrong_owner.public_key(),
        )


def test_primary_review_is_signed_and_keeps_authority_closed() -> None:
    packet = _build_packet()
    owner, reviewer, _, reviewer_policy, _ = _trust_pair()
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=_ANSWER_KEY,
        spec=_approved_review_spec(),
        reviewer_private_key=reviewer,
        trust_policy=reviewer_policy,
        owner_public_key=owner.public_key(),
    )

    verified = verify_web4_benchmark_human_review(
        review=review,
        packet=packet,
        answer_key_path=_ANSWER_KEY,
        reviewer_public_key=reviewer.public_key(),
        trust_policy=reviewer_policy,
        owner_public_key=owner.public_key(),
    )
    assert verified.human_reviewed is True
    assert verified.contains_answer_key is False
    assert verified.training_authorization is False
    assert verified.evaluation_authorization is False
    assert verified.promotion_eligible is False
    serialized = json.dumps(verified.model_dump(mode="json"), sort_keys=True)
    assert "REJECT_UNAUTHORIZED_WIDENING" not in serialized
    assert "must_request_new_grant_for_wider_scope" not in serialized


def test_primary_review_rejects_wrong_trust_role() -> None:
    packet = _build_packet()
    owner, _, adjudicator, _, adjudicator_policy = _trust_pair()

    with pytest.raises(ValueError, match="PRIMARY_REVIEWER"):
        sign_web4_benchmark_human_review(
            packet=packet,
            answer_key_path=_ANSWER_KEY,
            spec=_approved_review_spec(),
            reviewer_private_key=adjudicator,
            trust_policy=adjudicator_policy,
            owner_public_key=owner.public_key(),
        )


def test_primary_review_rejects_answer_key_drift(tmp_path: Path) -> None:
    packet = _build_packet()
    owner, reviewer, _, reviewer_policy, _ = _trust_pair()
    tampered = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    tampered["expected"]["authority_status"] = "STANDARD"
    answer_path = tmp_path / "answer-key.json"
    answer_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="drifted after intake"):
        sign_web4_benchmark_human_review(
            packet=packet,
            answer_key_path=answer_path,
            spec=_approved_review_spec(),
            reviewer_private_key=reviewer,
            trust_policy=reviewer_policy,
            owner_public_key=owner.public_key(),
        )


def test_rehashed_signature_tampering_is_rejected_cryptographically() -> None:
    packet = _build_packet()
    owner, reviewer, _, reviewer_policy, _ = _trust_pair()
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=_ANSWER_KEY,
        spec=_approved_review_spec(),
        reviewer_private_key=reviewer,
        trust_policy=reviewer_policy,
        owner_public_key=owner.public_key(),
    )
    tampered = review.model_dump(mode="json")
    signature = bytearray(base64.b64decode(tampered["signature_base64"]))
    signature[0] ^= 1
    tampered["signature_base64"] = base64.b64encode(bytes(signature)).decode("ascii")
    tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = _digest(tampered)
    rehashed = Web4BenchmarkHumanReview.model_validate(tampered)

    with pytest.raises(ValueError, match="signature verification failed"):
        verify_web4_benchmark_human_review(
            review=rehashed,
            packet=packet,
            answer_key_path=_ANSWER_KEY,
            reviewer_public_key=reviewer.public_key(),
            trust_policy=reviewer_policy,
            owner_public_key=owner.public_key(),
        )


def test_adjudication_requires_distinct_key_and_identity() -> None:
    packet = _build_packet()
    owner = Ed25519PrivateKey.generate()
    shared = Ed25519PrivateKey.generate()
    reviewer_policy = build_web4_reviewer_trust_policy(
        shared.public_key(),
        owner,
        policy_id="web4-shared-reviewer-v1",
        role=Web4ReviewRole.PRIMARY_REVIEWER,
        delegate_id="reviewer-fixture",
    )
    adjudicator_policy = build_web4_reviewer_trust_policy(
        shared.public_key(),
        owner,
        policy_id="web4-shared-adjudicator-v1",
        role=Web4ReviewRole.ADJUDICATOR,
        delegate_id="adjudicator-fixture",
    )
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=_ANSWER_KEY,
        spec=_approved_review_spec(),
        reviewer_private_key=shared,
        trust_policy=reviewer_policy,
        owner_public_key=owner.public_key(),
    )

    with pytest.raises(ValueError, match="different key"):
        adjudicate_web4_benchmark_review(
            review=review,
            packet=packet,
            answer_key_path=_ANSWER_KEY,
            reviewer_public_key=shared.public_key(),
            reviewer_trust_policy=reviewer_policy,
            adjudication_spec=_confirm_spec(),
            adjudicator_private_key=shared,
            adjudicator_trust_policy=adjudicator_policy,
            owner_public_key=owner.public_key(),
        )


def test_confirmed_holdout_becomes_release_candidate_but_not_eval_authorized(
    tmp_path: Path,
) -> None:
    packet, answer_path = _find_holdout_packet(tmp_path)
    owner, reviewer, adjudicator, reviewer_policy, adjudicator_policy = _trust_pair()
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=answer_path,
        spec=_approved_review_spec(),
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
        adjudication_spec=_confirm_spec(),
        adjudicator_private_key=adjudicator,
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=owner.public_key(),
    )

    verified = verify_web4_benchmark_adjudication(
        adjudication=adjudication,
        review=review,
        packet=packet,
        answer_key_path=answer_path,
        reviewer_public_key=reviewer.public_key(),
        reviewer_trust_policy=reviewer_policy,
        adjudicator_public_key=adjudicator.public_key(),
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=owner.public_key(),
    )
    assert verified.final_review_approved is True
    assert verified.eligible_for_signed_holdout_release is True
    assert verified.contains_answer_key is False
    assert verified.training_authorization is False
    assert verified.evaluation_authorization is False
    assert verified.promotion_eligible is False


def test_disputed_primary_review_never_becomes_release_candidate(tmp_path: Path) -> None:
    packet, answer_path = _find_holdout_packet(tmp_path)
    owner, reviewer, adjudicator, reviewer_policy, adjudicator_policy = _trust_pair()
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=answer_path,
        spec=_approved_review_spec(),
        reviewer_private_key=reviewer,
        trust_policy=reviewer_policy,
        owner_public_key=owner.public_key(),
    )
    dispute = Web4BenchmarkAdjudicationSpec(
        adjudicator_id="adjudicator-fixture",
        decision=Web4AdjudicationDecision.DISPUTE_PRIMARY,
        rationale="Independent adjudication found a material disagreement requiring re-review.",
    )
    adjudication = adjudicate_web4_benchmark_review(
        review=review,
        packet=packet,
        answer_key_path=answer_path,
        reviewer_public_key=reviewer.public_key(),
        reviewer_trust_policy=reviewer_policy,
        adjudication_spec=dispute,
        adjudicator_private_key=adjudicator,
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=owner.public_key(),
    )

    assert adjudication.final_review_approved is False
    assert adjudication.eligible_for_signed_holdout_release is False
    assert adjudication.evaluation_authorization is False


def test_adjudication_model_rejects_release_eligibility_tampering(tmp_path: Path) -> None:
    packet, answer_path = _find_holdout_packet(tmp_path)
    owner, reviewer, adjudicator, reviewer_policy, adjudicator_policy = _trust_pair()
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=answer_path,
        spec=_approved_review_spec(),
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
        adjudication_spec=_confirm_spec(),
        adjudicator_private_key=adjudicator,
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=owner.public_key(),
    )
    tampered = adjudication.model_dump(mode="json")
    tampered["eligible_for_signed_holdout_release"] = False
    tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = _digest(tampered)

    with pytest.raises(ValueError, match="release eligibility is inconsistent"):
        Web4BenchmarkAdjudication.model_validate(tampered)
