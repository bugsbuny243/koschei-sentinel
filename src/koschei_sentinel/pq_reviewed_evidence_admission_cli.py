from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.pq_evidence_review import (
    load_pq_owner_public_key,
    load_pq_reviewer_public_key,
)
from koschei_sentinel.pq_reviewed_evidence_admission import (
    build_pq_reviewed_evidence_catalog_entry,
    load_pq_reviewed_evidence_catalog_entry,
    verify_pq_reviewed_evidence_catalog_entry,
    write_pq_reviewed_evidence_catalog_entry,
)

_DEFAULT_ADMISSION_POLICY = Path("configs/corpus/pq-reviewed-evidence-admission.v1.json")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=_DEFAULT_ADMISSION_POLICY)
    parser.add_argument("--review-proof", type=Path, required=True)
    parser.add_argument("--trust-policy", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--snapshot-receipt", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--watch", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--reviewer-public-key", type=Path, required=True)
    parser.add_argument("--owner-public-key", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-pq-reviewed-catalog-admit",
        description=(
            "Admit VERIFIED PQ evidence into the immutable reviewed research catalog "
            "without granting dataset, split, evaluation, Gold, or training authorization."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    admit = subparsers.add_parser("admit")
    _add_common(admit)
    admit.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    _add_common(verify)
    verify.add_argument("--entry", type=Path, required=True)
    return parser


def _kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "admission_request_path": args.request,
        "admission_policy_path": args.policy,
        "review_proof_path": args.review_proof,
        "trust_policy_path": args.trust_policy,
        "claim_path": args.claim,
        "snapshot_receipt_path": args.snapshot_receipt,
        "snapshot_path": args.snapshot,
        "watch_registry_path": args.watch,
        "record_path": args.record,
        "materialization_receipt_path": args.materialization_receipt,
        "reviewer_public_key": load_pq_reviewer_public_key(args.reviewer_public_key),
        "owner_public_key": load_pq_owner_public_key(args.owner_public_key),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kwargs = _kwargs(args)
        if args.command == "admit":
            entry = build_pq_reviewed_evidence_catalog_entry(**kwargs)
            write_pq_reviewed_evidence_catalog_entry(entry, args.output)
        else:
            entry = load_pq_reviewed_evidence_catalog_entry(args.entry)
            entry = verify_pq_reviewed_evidence_catalog_entry(entry, **kwargs)
    except (OSError, TypeError, ValueError) as exc:
        print(f"PQ reviewed catalog admission rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
