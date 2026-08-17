from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.multi_incident_cyber_range import MultiIncidentCyberRangeScenario
from koschei_sentinel.multi_incident_cyber_range_suite import (
    MultiIncidentCyberRangeGatePolicy,
    run_multi_incident_cyber_range_suite,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run component-scoped Sentinel multi-incident cyber-range promotion scenarios"
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help="Multi-incident scenario JSON; repeat to build a suite",
    )
    parser.add_argument("--policy", help="Optional multi-incident gate policy JSON")
    parser.add_argument("--output", required=True, help="Suite report JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = [
            MultiIncidentCyberRangeScenario.model_validate_json(
                Path(path).read_text(encoding="utf-8")
            )
            for path in args.scenario
        ]
        policy = (
            MultiIncidentCyberRangeGatePolicy.model_validate_json(
                Path(args.policy).read_text(encoding="utf-8")
            )
            if args.policy
            else MultiIncidentCyberRangeGatePolicy()
        )
        report = run_multi_incident_cyber_range_suite(scenarios, policy=policy)
        payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if report.passed else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-multi-cyber-range: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
