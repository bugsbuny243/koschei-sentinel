from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.promotion import load_owner_private_key, load_owner_public_key
from koschei_sentinel.training import atomic_write
from koschei_sentinel.web4_benchmark_intake import Web4BenchmarkIntakePacket
from koschei_sentinel.web4_benchmark_review import (
    Web4BenchmarkAdjudicationSpec,
    Web4BenchmarkHumanReview,
    Web4BenchmarkHumanReviewSpec,
    adjudicate_web4_benchmark_review,
    sign_web4_benchmark_human_review,
)
from koschei_sentinel.web4_reviewer_trust import (
    Web4ReviewerTrustPolicy,
    Web4ReviewRole,
    build_web4_reviewer_trust_policy,
    load_web4_review_private_key,
    load_web4_review_public_key,
    load_web4_reviewer_trust_policy,
    write_web4_reviewer_trust_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-review-ops",
        description=(
            "Create owner-rooted Web4 reviewer trust, signed primary review, and "
            "independent adjudication artifacts without network or model execution."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    trust = commands.add_parser("trust")
    trust.add_argument("--delegate-public-key", required=True)
    trust.add_argument("--owner-private-key", required=True)
    trust.add_argument("--policy-id", required=True)
    trust.add_argument(
        "--role",
        required=True,
        choices=[role.value for role in Web4ReviewRole],
    )
    trust.add_argument("--delegate-id", required=True)
    trust.add_argument("--output", required=True)

    review = commands.add_parser("review")
    review.add_argument("--packet", required=True)
    review.add_argument("--answer-key", required=True)
    review.add_argument("--spec", required=True)
    review.add_argument("--reviewer-private-key", required=True)
    review.add_argument("--reviewer-trust-policy", required=True)
    review.add_argument("--owner-public-key", required=True)
    review.add_argument("--output", required=True)

    adjudicate = commands.add_parser("adjudicate")
    adjudicate.add_argument("--packet", required=True)
    adjudicate.add_argument("--answer-key", required=True)
    adjudicate.add_argument("--review", required=True)
    adjudicate.add_argument("--reviewer-public-key", required=True)
    adjudicate.add_argument("--reviewer-trust-policy", required=True)
    adjudicate.add_argument("--spec", required=True)
    adjudicate.add_argument("--adjudicator-private-key", required=True)
    adjudicate.add_argument("--adjudicator-trust-policy", required=True)
    adjudicate.add_argument("--owner-public-key", required=True)
    adjudicate.add_argument("--output", required=True)
    return parser


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate


def _preflight_output(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Web4 operator output already exists: {destination}")
    current = destination.parent
    while True:
        if current.is_symlink():
            raise ValueError("Web4 operator output path must not traverse symlink directories")
        if current.parent == current:
            break
        current = current.parent
    return destination


def _load_model(path: str | Path, model_type, label: str):
    candidate = _regular_file(path, label)
    try:
        return model_type.model_validate_json(candidate.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc


def _write_model(artifact, destination: Path) -> None:
    payload = json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(destination, payload)


def _trust(args: argparse.Namespace) -> dict[str, object]:
    destination = _preflight_output(args.output)
    delegate_key_path = _regular_file(args.delegate_public_key, "delegate public key")
    owner_key_path = _regular_file(args.owner_private_key, "owner private key")
    delegate_key = load_web4_review_public_key(delegate_key_path)
    owner_key = load_owner_private_key(owner_key_path)
    artifact = build_web4_reviewer_trust_policy(
        delegate_key,
        owner_key,
        policy_id=args.policy_id,
        role=Web4ReviewRole(args.role),
        delegate_id=args.delegate_id,
    )
    write_web4_reviewer_trust_policy(artifact, destination)
    return {
        "ok": True,
        "command": "trust",
        "policy_id": artifact.policy_id,
        "role": artifact.role.value,
        "delegate_id": artifact.delegate_id,
        "delegate_key_fingerprint": artifact.delegate_key_fingerprint,
        "owner_key_fingerprint": artifact.owner_key_fingerprint,
        "policy_digest": artifact.policy_digest,
        "owner_signature_verified": artifact.owner_signature_verified,
        "training_authorization_allowed": artifact.training_authorization_allowed,
        "evaluation_authorization_allowed": artifact.evaluation_authorization_allowed,
        "production_deployment_allowed": artifact.production_deployment_allowed,
        "output": str(destination),
    }


def _review(args: argparse.Namespace) -> dict[str, object]:
    destination = _preflight_output(args.output)
    packet = _load_model(args.packet, Web4BenchmarkIntakePacket, "Web4 intake packet")
    answer_key = _regular_file(args.answer_key, "Web4 answer key")
    spec = _load_model(args.spec, Web4BenchmarkHumanReviewSpec, "Web4 human review spec")
    reviewer_key_path = _regular_file(args.reviewer_private_key, "reviewer private key")
    trust_policy_path = _regular_file(args.reviewer_trust_policy, "reviewer trust policy")
    owner_key_path = _regular_file(args.owner_public_key, "owner public key")
    reviewer_key = load_web4_review_private_key(reviewer_key_path)
    trust_policy = load_web4_reviewer_trust_policy(trust_policy_path)
    owner_key = load_owner_public_key(owner_key_path)
    artifact = sign_web4_benchmark_human_review(
        packet=packet,
        answer_key_path=answer_key,
        spec=spec,
        reviewer_private_key=reviewer_key,
        trust_policy=trust_policy,
        owner_public_key=owner_key,
    )
    _write_model(artifact, destination)
    return {
        "ok": True,
        "command": "review",
        "case_id": artifact.case_id,
        "split": artifact.split.value,
        "decision": artifact.decision.value,
        "reviewer_id": artifact.reviewer_id,
        "reviewer_key_fingerprint": artifact.reviewer_key_fingerprint,
        "review_sha256": artifact.review_sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "signature_verified": artifact.signature_verified,
        "contains_answer_key": artifact.contains_answer_key,
        "training_authorization": artifact.training_authorization,
        "evaluation_authorization": artifact.evaluation_authorization,
        "promotion_eligible": artifact.promotion_eligible,
        "output": str(destination),
    }


def _adjudicate(args: argparse.Namespace) -> dict[str, object]:
    destination = _preflight_output(args.output)
    packet = _load_model(args.packet, Web4BenchmarkIntakePacket, "Web4 intake packet")
    answer_key = _regular_file(args.answer_key, "Web4 answer key")
    review = _load_model(args.review, Web4BenchmarkHumanReview, "Web4 human review")
    reviewer_public_key_path = _regular_file(args.reviewer_public_key, "reviewer public key")
    reviewer_policy_path = _regular_file(args.reviewer_trust_policy, "reviewer trust policy")
    spec = _load_model(args.spec, Web4BenchmarkAdjudicationSpec, "Web4 adjudication spec")
    adjudicator_key_path = _regular_file(
        args.adjudicator_private_key,
        "adjudicator private key",
    )
    adjudicator_policy_path = _regular_file(
        args.adjudicator_trust_policy,
        "adjudicator trust policy",
    )
    owner_key_path = _regular_file(args.owner_public_key, "owner public key")

    reviewer_public_key = load_web4_review_public_key(reviewer_public_key_path)
    reviewer_policy = load_web4_reviewer_trust_policy(reviewer_policy_path)
    adjudicator_private_key = load_web4_review_private_key(adjudicator_key_path)
    adjudicator_policy = load_web4_reviewer_trust_policy(adjudicator_policy_path)
    owner_public_key = load_owner_public_key(owner_key_path)
    artifact = adjudicate_web4_benchmark_review(
        review=review,
        packet=packet,
        answer_key_path=answer_key,
        reviewer_public_key=reviewer_public_key,
        reviewer_trust_policy=reviewer_policy,
        adjudication_spec=spec,
        adjudicator_private_key=adjudicator_private_key,
        adjudicator_trust_policy=adjudicator_policy,
        owner_public_key=owner_public_key,
    )
    _write_model(artifact, destination)
    return {
        "ok": True,
        "command": "adjudicate",
        "case_id": artifact.case_id,
        "split": artifact.split.value,
        "decision": artifact.decision.value,
        "adjudicator_id": artifact.adjudicator_id,
        "adjudicator_key_fingerprint": artifact.adjudicator_key_fingerprint,
        "adjudication_sha256": artifact.adjudication_sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "signature_verified": artifact.signature_verified,
        "independent_adjudication": artifact.independent_adjudication,
        "final_review_approved": artifact.final_review_approved,
        "eligible_for_signed_holdout_release": artifact.eligible_for_signed_holdout_release,
        "contains_answer_key": artifact.contains_answer_key,
        "training_authorization": artifact.training_authorization,
        "evaluation_authorization": artifact.evaluation_authorization,
        "promotion_eligible": artifact.promotion_eligible,
        "output": str(destination),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "trust":
            result = _trust(args)
        elif args.command == "review":
            result = _review(args)
        else:
            result = _adjudicate(args)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-web4-review-ops: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
