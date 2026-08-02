from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.adapters import load_candidate_registry
from koschei_sentinel.benchmark import BenchmarkThresholds, load_benchmark_suite
from koschei_sentinel.matrix import run_comparison, write_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-compare",
        description="Run Sentinel candidates through one deterministic comparison matrix.",
    )
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--allow-local-network", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-authority-score", type=float, default=1.0)
    parser.add_argument("--min-grounding-score", type=float, default=1.0)
    parser.add_argument("--min-abstention-score", type=float, default=1.0)
    parser.add_argument("--min-privacy-score", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_local_network and not args.allow_network:
        raise SystemExit("--allow-local-network requires --allow-network")
    try:
        suite = load_benchmark_suite(args.suite)
        registry = load_candidate_registry(args.registry)
        run = run_comparison(
            suite,
            registry,
            registry_base_dir=args.registry.parent,
            thresholds=BenchmarkThresholds(
                min_case_pass_rate=args.min_case_pass_rate,
                min_authority_score=args.min_authority_score,
                min_grounding_score=args.min_grounding_score,
                min_abstention_score=args.min_abstention_score,
                min_privacy_score=args.min_privacy_score,
            ),
            allow_network=args.allow_network,
            allow_local_network=args.allow_local_network,
        )
        if args.output_dir is not None:
            write_comparison(run, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"comparison rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(run.matrix.model_dump(mode="json"), indent=2, sort_keys=True))
    if args.require_all:
        return 0 if run.matrix.all_candidates_passed else 3
    return 0 if run.matrix.any_candidate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
