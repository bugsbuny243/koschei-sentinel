from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.agentic_security_review import load_and_validate_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an agentic-security benchmark review bundle.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args()

    source, review = load_and_validate_review(args.source, args.review)
    counts = {"accepted": 0, "rejected": 0, "needs_rewrite": 0}
    for record in review.records:
        counts[record.decision] += 1

    print(json.dumps({
        "valid": True,
        "source_cases": source.total_cases,
        "reviewed_cases": len(review.records),
        "decisions": counts,
        "accepted_are_gold": review.accepted_are_gold,
        "automatic_gold_promotion_allowed": review.automatic_gold_promotion_allowed,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
