from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_range import CyberRangeScenario
from koschei_sentinel.cyber_range_suite import (
    CyberRangeGatePolicy,
    run_cyber_range_suite,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Koschei Sentinel active-defense cyber-range scenarios"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help="Cyber Range scenario JSON file; repeat for a suite",
    )
    parser.add_argument("--policy", help="Optional Cyber Range gate policy JSON file")
    parser.add_argument("--output", help="Optional suite report JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = [
            CyberRangeScenario.model_validate_json(Path(path).read_text(encoding="utf-8"))
            for path in args.scenario
        ]
        policy = (
            CyberRangeGatePolicy.model_validate_json(
                Path(args.policy).read_text(encoding="utf-8")
            )
            if args.policy
            else CyberRangeGatePolicy()
        )
        report = run_cyber_range_suite(scenarios, policy=policy)
        payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0 if report.passed else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-range: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
