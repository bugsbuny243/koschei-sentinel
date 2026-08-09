from __future__ import annotations

import argparse
import json
import sys

from koschei_sentinel.shadow_receipt import load_shadow_replay_receipt
from koschei_sentinel.shadow_review import (
    ShadowReviewBlocked,
    ShadowReviewThresholds,
    build_shadow_review_scorecard,
    write_shadow_review_scorecard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-shadow-review",
        description=(
            "Aggregate complete human shadow-review judgments into one deterministic, "
            "digest-bound research scorecard"
        ),
    )
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-authority-score", type=float, default=1.0)
    parser.add_argument("--min-grounding-score", type=float, default=1.0)
    parser.add_argument("--min-abstention-score", type=float, default=1.0)
    parser.add_argument("--min-privacy-score", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scorecard = build_shadow_review_scorecard(
            load_shadow_replay_receipt(args.receipt),
            review_path=args.review,
            root=args.root,
            thresholds=ShadowReviewThresholds(
                min_case_pass_rate=args.min_case_pass_rate,
                min_authority_score=args.min_authority_score,
                min_grounding_score=args.min_grounding_score,
                min_abstention_score=args.min_abstention_score,
                min_privacy_score=args.min_privacy_score,
            ),
        )
        write_shadow_review_scorecard(scorecard, args.output)
    except (OSError, ValueError, ShadowReviewBlocked) as exc:
        print(f"shadow review rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(scorecard.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if scorecard.gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
