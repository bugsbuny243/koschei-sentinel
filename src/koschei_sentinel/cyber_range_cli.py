from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--reports-dir",
        help="Optional directory for individual immutable Cyber Range scenario reports",
    )
    return parser


def _report_filename(index: int, scenario_id: str) -> str:
    digest = hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:16]
    return f"{index:04d}-{digest}.json"


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
        elif not args.reports_dir:
            print(payload, end="")

        if args.reports_dir:
            reports_dir = Path(args.reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            for index, scenario_report in enumerate(report.scenario_reports, 1):
                report_payload = (
                    json.dumps(
                        scenario_report.model_dump(mode="json"),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                report_path = reports_dir / _report_filename(index, scenario_report.scenario_id)
                report_path.write_text(report_payload, encoding="utf-8")

        return 0 if report.passed else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-range: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
