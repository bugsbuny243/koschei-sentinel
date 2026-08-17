from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.defense_load_range import (
    DefenseLoadGatePolicy,
    run_defense_load_range,
)
from koschei_sentinel.defense_resource_scheduler import DefenseResourcePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stress-test Sentinel defense scheduling under constrained resources"
    )
    parser.add_argument("--plan", required=True, help="Assured multi-incident defense plan JSON")
    parser.add_argument("--resource-policy", required=True, help="Defense resource policy JSON")
    parser.add_argument("--gate-policy", required=True, help="Defense load gate policy JSON")
    parser.add_argument("--output", required=True, help="Defense load range report JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = AssuredMultiIncidentDefensePlan.model_validate_json(
            Path(args.plan).read_text(encoding="utf-8")
        )
        resource_policy = DefenseResourcePolicy.model_validate_json(
            Path(args.resource_policy).read_text(encoding="utf-8")
        )
        gate_policy = DefenseLoadGatePolicy.model_validate_json(
            Path(args.gate_policy).read_text(encoding="utf-8")
        )
        report = run_defense_load_range(
            plan,
            resource_policy=resource_policy,
            gate_policy=gate_policy,
        )
        payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if report.passed else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-load-range: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
