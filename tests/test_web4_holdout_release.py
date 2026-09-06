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
    Web4BenchmarkAdjudicationSpec,
    Web4BenchmarkHumanReviewSpec,
    Web4HumanReviewDecision,
    adjudicate_web4_benchmark_review,
    sign_web4_benchmark_human_review,
)
from koschei_sentinel.web4_holdout_release import (
    Web4HoldoutRelease,
    Web4HoldoutReleaseMaterial,
    build_web4_holdout_release,
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


def _find_holdout_packet(tmp_path: Path) -> tuple[Web4BenchmarkIntakePacket, Path]:
    proposal_template = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    answer_template = json.loads(_ANSWER_KEY.read_text(encoding="utf-8"))
    for index in range(1, 200):
        case_id = f"fixture-web4-release-{index:03d}"
        proposal = dict(proposal_template)
        proposal["case_id"] = case_id
        answer = dict(answer_template)
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


def _material(
    tmp_path: Path,
    *,
    adjudication_decision: Web4AdjudicationDecision = Web4AdjudicationDecision.CONFIRM_PRIMARY,
):
    packet, answer_path = _find_holdout_packet(tmp_path)
    owner = Ed25519PrivateKey.generate()
    reviewer = Ed25519PrivateKey.generate()
    adjudicator = Ed25519PrivateKey.generate()
    reviewer_policy = build_web4_reviewer_trust_policy(
        reviewer.public_key(),
        owner,
        policy_id="web4-release-reviewer-v1",
        role=Web4ReviewRole.PRIMARY_REVIEWER,
        delegate_id="release-reviewer",
    )
    adjudicator_policy = build_web4_reviewer_trust_policy(
        adjudicator.public_key(),
        owner,
        policy_id="web4-release-adjudicator-v1",
        role=Web4ReviewRole.ADJUDICATOR,
        delegate_id="release-adjudicator",
    )
    review = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=answer_path,
        spec=Web4BenchmarkHumanReviewSpec(
            reviewer_id="release-reviewer",
            decision=Web4HumanReviewDecision.APPROVE,
            rationale="Primary reviewer verified provenance, answer key, family, and authority status.",
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
            adjudicator_id="release-adjudicator",
            decision=adjudication_decision,
            rationale="Independent adjudicator checked the isolated answer key and source bindings.",
            independently_checked_answer_key=(
                adjudication_decision is Web4AdjudicationDecision.CONFIRM_PRIMARY
            ),
            independently_checked_source_bindings=(
                adjudication_decision is Web4AdjudicationDecision.CONFIRM_PRIMARY
            ),
            independently_checked_review_reasoning=(
                adjudication_decision is Web4AdjudicationDecision.CONFIRM_PRIMARY
            ),
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
    return owner, material


def test_owner_signed_holdout_release_authorizes_research_eval_only(tmp_path: Path) -> None:
    owner, material = _material(tmp_path)
    release = build_web4_holdout_release(
        release_id="web4-holdout-fixture-v1",
        materials=[material],
        benchmark_policy_path=_BENCHMARK,
        owner_private_key=owner,
    )

    verified = verify_web4_holdout_release(
        release,
        owner_public_key=owner.public_key(),
        benchmark_policy_path=_BENCHMARK,
        materials=[material],
    )
    assert verified.case_count == 1
    assert verified.cases[0].split == "HOLDOUT"
    assert verified.research_evaluation_authorization is True
    assert verified.training_authorization is False
    assert verified.promotion_eligible is False
    assert verified.production_activation_allowed is False
    assert verified.cases[0].contains_answer_key is False
    serialized = json.dumps(verified.model_dump(mode="json"), sort_keys=True)
    assert "REJECT_UNAUTHORIZED_WIDENING" not in serialized
    assert "must_request_new_grant_for_wider_scope" not in serialized


def test_non_holdout_packet_is_rejected(tmp_path: Path) -> None:
    packet = _build_packet()
    assert packet.split is not Web4BenchmarkSplit.HOLDOUT
    owner, material = _material(tmp_path)
    material = Web4HoldoutReleaseMaterial(
        packet=packet,
        review=material.review,
        adjudication=material.adjudication,
        answer_key_path=material.answer_key_path,
        reviewer_public_key=material.reviewer_public_key,
        reviewer_trust_policy=material.reviewer_trust_policy,
        adjudicator_public_key=material.adjudicator_public_key,
        adjudicator_trust_policy=material.adjudicator_trust_policy,
    )

    with pytest.raises(ValueError, match="non-HOLDOUT"):
        build_web4_holdout_release(
            release_id="web4-non-holdout-v1",
            materials=[material],
            benchmark_policy_path=_BENCHMARK,
            owner_private_key=owner,
        )


def test_disputed_adjudication_is_rejected_from_release(tmp_path: Path) -> None:
    owner, material = _material(
        tmp_path,
        adjudication_decision=Web4AdjudicationDecision.DISPUTE_PRIMARY,
    )

    with pytest.raises(ValueError, match="final approved review"):
        build_web4_holdout_release(
            release_id="web4-disputed-v1",
            materials=[material],
            benchmark_policy_path=_BENCHMARK,
            owner_private_key=owner,
        )


def test_wrong_owner_public_key_cannot_verify_release(tmp_path: Path) -> None:
    owner, material = _material(tmp_path)
    release = build_web4_holdout_release(
        release_id="web4-owner-check-v1",
        materials=[material],
        benchmark_policy_path=_BENCHMARK,
        owner_private_key=owner,
    )
    wrong_owner = Ed25519PrivateKey.generate()

    with pytest.raises(ValueError, match="owner key fingerprint differs"):
        verify_web4_holdout_release(
            release,
            owner_public_key=wrong_owner.public_key(),
            benchmark_policy_path=_BENCHMARK,
        )


def test_rehashed_owner_signature_tampering_is_rejected(tmp_path: Path) -> None:
    owner, material = _material(tmp_path)
    release = build_web4_holdout_release(
        release_id="web4-signature-check-v1",
        materials=[material],
        benchmark_policy_path=_BENCHMARK,
        owner_private_key=owner,
    )
    tampered = release.model_dump(mode="json")
    signature = bytearray(base64.b64decode(tampered["owner_signature_base64"]))
    signature[-1] ^= 1
    tampered["owner_signature_base64"] = base64.b64encode(bytes(signature)).decode("ascii")
    tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = _digest(tampered)
    rehashed = Web4HoldoutRelease.model_validate(tampered)

    with pytest.raises(ValueError, match="owner signature verification failed"):
        verify_web4_holdout_release(
            rehashed,
            owner_public_key=owner.public_key(),
            benchmark_policy_path=_BENCHMARK,
        )


def test_release_replay_detects_different_review_chain(tmp_path: Path) -> None:
    owner, material = _material(tmp_path)
    release = build_web4_holdout_release(
        release_id="web4-replay-check-v1",
        materials=[material],
        benchmark_policy_path=_BENCHMARK,
        owner_private_key=owner,
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_owner, other_material = _material(other_dir)
    assert other_owner.public_key() != owner.public_key()

    with pytest.raises(ValueError):
        verify_web4_holdout_release(
            release,
            owner_public_key=owner.public_key(),
            benchmark_policy_path=_BENCHMARK,
            materials=[other_material],
        )


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    owner, material = _material(tmp_path)

    with pytest.raises(ValueError, match="duplicate case IDs"):
        build_web4_holdout_release(
            release_id="web4-duplicate-v1",
            materials=[material, material],
            benchmark_policy_path=_BENCHMARK,
            owner_private_key=owner,
        )
