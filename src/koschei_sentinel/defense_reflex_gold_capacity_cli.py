from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.defense_reflex_gold_capacity import (
    build_gold_review_capacity_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the owner-trusted signed human-review artifact set and measure "
            "the actual deterministic TRAIN/VALIDATION/HOLDOUT capacity"
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
    parser.add_argument("--minimum-train-cases", type=int, default=1)
    parser.add_argument("--minimum-validation-cases", type=int, default=1)
    parser.add_argument("--minimum-holdout-cases", type=int, default=50)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_gold_review_capacity_report(
            scenario_dir=args.scenarios,
            packet_dir=args.packets,
            review_dir=args.reviews,
            signature_dir=args.signatures,
            split_policy_path=args.split_policy,
            reviewer_public_key_path=args.reviewer_public_key,
            reviewer_trust_policy_path=args.reviewer_trust_policy,
            owner_public_key_path=args.owner_public_key,
            minimum_train_cases=args.minimum_train_cases,
            minimum_validation_cases=args.minimum_validation_cases,
            minimum_holdout_cases=args.minimum_holdout_cases,
        )
        payload = json.dumps(
            report.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ) + "\n"
        if args.output:
            destination = Path(args.output)
            if destination.exists():
                raise FileExistsError(f"Gold capacity output already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if report.ready_for_release else 1
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"sentinel-defense-reflex-gold-capacity: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
