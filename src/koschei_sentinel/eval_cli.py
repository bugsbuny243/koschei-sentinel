from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.benchmark import (
    BenchmarkThresholds,
    baseline_predictions,
    evaluate_benchmark,
    load_benchmark_suite,
    load_predictions,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-eval",
        description="Run the deterministic Sentinel model benchmark quality gate.",
    )
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--candidate", default="sentinel-baseline-v0.3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-authority-score", type=float, default=1.0)
    parser.add_argument("--min-grounding-score", type=float, default=1.0)
    parser.add_argument("--min-abstention-score", type=float, default=1.0)
    parser.add_argument("--min-privacy-score", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        suite = load_benchmark_suite(args.suite)
        predictions = (
            load_predictions(args.predictions)
            if args.predictions is not None
            else baseline_predictions(suite, candidate=args.candidate)
        )
        report = evaluate_benchmark(
            suite,
            predictions,
            thresholds=BenchmarkThresholds(
                min_case_pass_rate=args.min_case_pass_rate,
                min_authority_score=args.min_authority_score,
                min_grounding_score=args.min_grounding_score,
                min_abstention_score=args.min_abstention_score,
                min_privacy_score=args.min_privacy_score,
            ),
        )
        if args.output is not None:
            write_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"benchmark rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
