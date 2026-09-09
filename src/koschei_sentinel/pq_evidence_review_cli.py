from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.pq_evidence_review import (
    build_pq_evidence_review_proof,
    build_pq_evidence_reviewer_trust_policy,
    load_pq_evidence_review_proof,
    load_pq_evidence_reviewer_trust_policy,
    load_pq_owner_private_key,
    load_pq_owner_public_key,
    load_pq_reviewer_private_key,
    load_pq_reviewer_public_key,
    verify_pq_evidence_review_proof,
    verify_pq_evidence_reviewer_trust_policy,
    write_pq_evidence_review_proof,
    write_pq_evidence_reviewer_trust_policy,
)

_DEFAULT_WATCH = Path("configs/corpus/pq-network-watch.v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-pq-evidence-review",
        description=(
            "Owner-authorize PQ evidence reviewers and sign or verify immutable, "
            "non-training-authorizing PQ evidence review proofs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trust_create = subparsers.add_parser("trust-create")
    trust_create.add_argument("--policy-id", required=True)
    trust_create.add_argument("--reviewer-public-key", type=Path, required=True)
    trust_create.add_argument("--owner-private-key", type=Path, required=True)
    trust_create.add_argument("--output", type=Path, required=True)

    trust_verify = subparsers.add_parser("trust-verify")
    trust_verify.add_argument("--policy", type=Path, required=True)
    trust_verify.add_argument("--reviewer-public-key", type=Path, required=True)
    trust_verify.add_argument("--owner-public-key", type=Path, required=True)

    sign = subparsers.add_parser("sign")
    sign.add_argument("--review-input", type=Path, required=True)
    sign.add_argument("--claim", type=Path, required=True)
    sign.add_argument("--snapshot-receipt", type=Path, required=True)
    sign.add_argument("--snapshot", type=Path, required=True)
    sign.add_argument("--watch", type=Path, default=_DEFAULT_WATCH)
    sign.add_argument("--record", type=Path, required=True)
    sign.add_argument("--materialization-receipt", type=Path, required=True)
    sign.add_argument("--reviewer-private-key", type=Path, required=True)
    sign.add_argument("--trust-policy", type=Path, required=True)
    sign.add_argument("--owner-public-key", type=Path, required=True)
    sign.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--proof", type=Path, required=True)
    verify.add_argument("--claim", type=Path, required=True)
    verify.add_argument("--snapshot-receipt", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--watch", type=Path, default=_DEFAULT_WATCH)
    verify.add_argument("--record", type=Path, required=True)
    verify.add_argument("--materialization-receipt", type=Path, required=True)
    verify.add_argument("--reviewer-public-key", type=Path, required=True)
    verify.add_argument("--trust-policy", type=Path, required=True)
    verify.add_argument("--owner-public-key", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "trust-create":
            policy = build_pq_evidence_reviewer_trust_policy(
                reviewer_public_key=load_pq_reviewer_public_key(args.reviewer_public_key),
                owner_private_key=load_pq_owner_private_key(args.owner_private_key),
                policy_id=args.policy_id,
            )
            write_pq_evidence_reviewer_trust_policy(policy, args.output)
            payload = policy.model_dump(mode="json")
        elif args.command == "trust-verify":
            policy = load_pq_evidence_reviewer_trust_policy(args.policy)
            verified = verify_pq_evidence_reviewer_trust_policy(
                policy,
                reviewer_public_key=load_pq_reviewer_public_key(args.reviewer_public_key),
                owner_public_key=load_pq_owner_public_key(args.owner_public_key),
            )
            payload = verified.model_dump(mode="json")
        elif args.command == "sign":
            policy = load_pq_evidence_reviewer_trust_policy(args.trust_policy)
            proof = build_pq_evidence_review_proof(
                review_input_path=args.review_input,
                claim_path=args.claim,
                snapshot_receipt_path=args.snapshot_receipt,
                snapshot_path=args.snapshot,
                watch_registry_path=args.watch,
                record_path=args.record,
                materialization_receipt_path=args.materialization_receipt,
                reviewer_private_key=load_pq_reviewer_private_key(args.reviewer_private_key),
                trust_policy=policy,
                owner_public_key=load_pq_owner_public_key(args.owner_public_key),
            )
            write_pq_evidence_review_proof(proof, args.output)
            payload = proof.model_dump(mode="json")
        else:
            policy = load_pq_evidence_reviewer_trust_policy(args.trust_policy)
            proof = load_pq_evidence_review_proof(args.proof)
            verified = verify_pq_evidence_review_proof(
                proof,
                claim_path=args.claim,
                snapshot_receipt_path=args.snapshot_receipt,
                snapshot_path=args.snapshot,
                watch_registry_path=args.watch,
                record_path=args.record,
                materialization_receipt_path=args.materialization_receipt,
                reviewer_public_key=load_pq_reviewer_public_key(args.reviewer_public_key),
                trust_policy=policy,
                owner_public_key=load_pq_owner_public_key(args.owner_public_key),
            )
            payload = verified.model_dump(mode="json")
    except (OSError, TypeError, ValueError) as exc:
        print(f"PQ evidence review rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
