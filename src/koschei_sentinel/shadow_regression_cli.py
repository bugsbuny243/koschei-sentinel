from __future__ import annotations

import argparse
import json
import sys

from koschei_sentinel.shadow_receipt import load_shadow_replay_receipt
from koschei_sentinel.shadow_regression import (
    ShadowRegressionBlocked,
    ShadowRegressionTolerance,
    append_shadow_regression_history,
    build_shadow_regression_report,
    load_shadow_regression_history,
    write_shadow_regression_history,
    write_shadow_regression_report,
)
from koschei_sentinel.shadow_review import load_shadow_review_scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-shadow-regression",
        description="Compare two human-reviewed shadow runs on the same sealed replay dataset",
    )
    parser.add_argument("--baseline-scorecard", required=True)
    parser.add_argument("--baseline-receipt", required=True)
    parser.add_argument("--candidate-scorecard", required=True)
    parser.add_argument("--candidate-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-in")
    parser.add_argument("--history-out")
    parser.add_argument("--max-score-drop", type=float, default=0.0)
    parser.add_argument("--max-failed-case-increase", type=int, default=0)
    parser.add_argument("--max-followup-case-increase", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.history_in and not args.history_out:
            raise ValueError("--history-in requires --history-out")
        report = build_shadow_regression_report(
            load_shadow_review_scorecard(args.baseline_scorecard),
            load_shadow_replay_receipt(args.baseline_receipt),
            load_shadow_review_scorecard(args.candidate_scorecard),
            load_shadow_replay_receipt(args.candidate_receipt),
            tolerance=ShadowRegressionTolerance(
                max_score_drop=args.max_score_drop,
                max_failed_case_increase=args.max_failed_case_increase,
                max_followup_case_increase=args.max_followup_case_increase,
            ),
        )

        history = None
        if args.history_out:
            previous = load_shadow_regression_history(args.history_in) if args.history_in else None
            history = append_shadow_regression_history(report, previous)

        write_shadow_regression_report(report, args.output)
        if history is not None:
            write_shadow_regression_history(history, args.history_out)
    except (OSError, ValueError, ShadowRegressionBlocked) as exc:
        print(f"shadow regression rejected: {exc}", file=sys.stderr)
        return 2

    result = report.model_dump(mode="json")
    if history is not None:
        result["history_digest"] = history.history_digest
        result["history_entries"] = len(history.entries)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if report.regression_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
