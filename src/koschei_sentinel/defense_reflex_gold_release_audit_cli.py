from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.defense_reflex_gold_release_audit import (
    audit_gold_defense_release,
)
from koschei_sentinel.gold_review_signing import audit_gold_release_review_signatures
from koschei_sentinel.gold_reviewer_trust import load_trusted_reviewer_public_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit structural integrity and trusted Ed25519 human-review signatures "
            "for a Gold Defense Reflex release"
        )
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--reviewer-trust-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--output", help="Optional combined JSON audit report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reviewer_public_key = load_trusted_reviewer_public_key(
            reviewer_public_key_path=args.reviewer_public_key,
            trust_policy_path=args.reviewer_trust_policy,
            owner_public_key_path=args.owner_public_key,
        )
        structural = audit_gold_defense_release(args.release_dir)
        signature = audit_gold_release_review_signatures(
            args.release_dir,
            reviewer_public_key,
        )
        valid = structural.valid and signature.valid
        report = {
            "schema_version": "sentinel.gold-signed-release-audit.v1",
            "valid": valid,
            "structural_audit": structural.model_dump(mode="json"),
            "review_signature_audit": signature.model_dump(mode="json"),
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if valid else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-defense-reflex-gold-audit: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
