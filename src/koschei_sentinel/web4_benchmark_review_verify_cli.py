from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.promotion import load_owner_public_key
from koschei_sentinel.web4_benchmark_intake import Web4BenchmarkIntakePacket
from koschei_sentinel.web4_benchmark_review import (
    Web4BenchmarkAdjudication,
    Web4BenchmarkHumanReview,
    verify_web4_benchmark_adjudication,
)
from koschei_sentinel.web4_reviewer_trust import (
    Web4ReviewRole,
    load_trusted_web4_review_public_key,
    load_web4_reviewer_trust_policy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify owner-rooted Web4 human review and independent adjudication."
    )
    parser.add_argument("--packet", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--reviewer-trust-policy", required=True)
    parser.add_argument("--adjudication", required=True)
    parser.add_argument("--adjudicator-public-key", required=True)
    parser.add_argument("--adjudicator-trust-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    return parser


def _load_model(path: str, model_type, label: str):
    try:
        return model_type.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc


def main() -> int:
    args = _parser().parse_args()
    try:
        packet = _load_model(args.packet, Web4BenchmarkIntakePacket, "Web4 intake packet")
        review = _load_model(args.review, Web4BenchmarkHumanReview, "Web4 human review")
        adjudication = _load_model(
            args.adjudication,
            Web4BenchmarkAdjudication,
            "Web4 adjudication",
        )
        reviewer_policy = load_web4_reviewer_trust_policy(args.reviewer_trust_policy)
        adjudicator_policy = load_web4_reviewer_trust_policy(args.adjudicator_trust_policy)
        owner_public_key = load_owner_public_key(args.owner_public_key)
        reviewer_public_key = load_trusted_web4_review_public_key(
            delegate_public_key_path=args.reviewer_public_key,
            trust_policy_path=args.reviewer_trust_policy,
            owner_public_key_path=args.owner_public_key,
            required_role=Web4ReviewRole.PRIMARY_REVIEWER,
        )
        adjudicator_public_key = load_trusted_web4_review_public_key(
            delegate_public_key_path=args.adjudicator_public_key,
            trust_policy_path=args.adjudicator_trust_policy,
            owner_public_key_path=args.owner_public_key,
            required_role=Web4ReviewRole.ADJUDICATOR,
        )
        verified = verify_web4_benchmark_adjudication(
            adjudication=adjudication,
            review=review,
            packet=packet,
            answer_key_path=args.answer_key,
            reviewer_public_key=reviewer_public_key,
            reviewer_trust_policy=reviewer_policy,
            adjudicator_public_key=adjudicator_public_key,
            adjudicator_trust_policy=adjudicator_policy,
            owner_public_key=owner_public_key,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "schema_version": "sentinel.web4-review-verification.v1",
                "valid": True,
                "case_id": verified.case_id,
                "split": verified.split.value,
                "packet_sha256": verified.packet_sha256,
                "review_sha256": verified.review_sha256,
                "review_artifact_sha256": verified.review_artifact_sha256,
                "adjudication_sha256": verified.adjudication_sha256,
                "adjudication_artifact_sha256": verified.artifact_sha256,
                "reviewer_trust_policy_digest": review.reviewer_trust_policy_digest,
                "adjudicator_trust_policy_digest": verified.adjudicator_trust_policy_digest,
                "final_review_approved": verified.final_review_approved,
                "eligible_for_signed_holdout_release": verified.eligible_for_signed_holdout_release,
                "contains_answer_key": verified.contains_answer_key,
                "training_authorization": verified.training_authorization,
                "evaluation_authorization": verified.evaluation_authorization,
                "promotion_eligible": verified.promotion_eligible,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
