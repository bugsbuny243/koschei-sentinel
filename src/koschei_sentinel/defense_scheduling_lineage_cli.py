from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.attack_world_lines import AttackWorldLineTimeline
from koschei_sentinel.defense_resource_scheduler import DefenseSchedulerState
from koschei_sentinel.defense_scheduling_lineage import (
    build_world_line_scheduling_context,
    carry_scheduler_state_across_world_lines,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind defense scheduling to persistent attack world-line identity"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    context = sub.add_parser("context", help="Build a scheduling context for one timeline tick")
    context.add_argument("--plan", required=True)
    context.add_argument("--world-lines", required=True)
    context.add_argument("--tick", required=True, type=int)
    context.add_argument("--output", required=True)

    carry = sub.add_parser("carry", help="Carry scheduler wait age across adjacent timeline ticks")
    carry.add_argument("--state", required=True)
    carry.add_argument("--world-lines", required=True)
    carry.add_argument("--from-tick", required=True, type=int)
    carry.add_argument("--to-tick", required=True, type=int)
    carry.add_argument("--output", required=True)
    return parser


def _write(path: str, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        timeline = AttackWorldLineTimeline.model_validate_json(
            Path(args.world_lines).read_text(encoding="utf-8")
        )
        if args.command == "context":
            plan = AssuredMultiIncidentDefensePlan.model_validate_json(
                Path(args.plan).read_text(encoding="utf-8")
            )
            result = build_world_line_scheduling_context(
                plan,
                timeline,
                tick=args.tick,
            )
            _write(args.output, result.model_dump(mode="json"))
            return 0

        state = DefenseSchedulerState.model_validate_json(
            Path(args.state).read_text(encoding="utf-8")
        )
        result = carry_scheduler_state_across_world_lines(
            state,
            timeline,
            from_tick=args.from_tick,
            to_tick=args.to_tick,
        )
        _write(args.output, result.model_dump(mode="json"))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-lineage: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
