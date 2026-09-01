from __future__ import annotations

import argparse
import json

from koschei_sentinel.defense_reflex_gold_production_release import (
    build_gold_production_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a production Gold release only after the owner-trusted signed "
            "human-review set deterministically contains at least 50 HOLDOUT cases"
        )
    )
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--split-policy", required=True)
    parser.add_argument("--reviewer-public-key", required=True)
    parser.add_argument("--reviewer-trust-policy", required=True)
    parser.add_argument("--owner-public-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_gold_production_release(
            scenario_dir=args.scenarios,
            packet_dir=args.packets,
            review_dir=args.reviews,
            signature_dir=args.signatures,
            split_policy_path=args.split_policy,
            reviewer_public_key_path=args.reviewer_public_key,
            reviewer_trust_policy_path=args.reviewer_trust_policy,
            owner_public_key_path=args.owner_public_key,
            output_dir=args.output_dir,
            receipt_output=args.receipt_output,
        )
        payload = json.dumps(
            receipt.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ) + "\n"
        print(payload, end="")
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-defense-reflex-gold-production-release: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
