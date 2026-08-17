from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.defense_resource_scheduler import (
    DefenseResourcePolicy,
    DefenseSchedulerState,
    DefenseSchedulingContext,
    build_defense_resource_schedule,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Schedule one safe parallel defense wave from an assured multi-incident plan"
        )
    )
    parser.add_argument("--plan", required=True, help="Assured multi-incident defense plan JSON")
    parser.add_argument("--policy", help="Optional defense resource policy JSON")
    parser.add_argument("--state", help="Optional prior scheduler state JSON")
    parser.add_argument(
        "--context",
        help="Optional world-line-aware defense scheduling context JSON",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = AssuredMultiIncidentDefensePlan.model_validate_json(
            Path(args.plan).read_text(encoding="utf-8")
        )
        policy = (
            DefenseResourcePolicy.model_validate_json(
                Path(args.policy).read_text(encoding="utf-8")
            )
            if args.policy
            else DefenseResourcePolicy()
        )
        state = (
            DefenseSchedulerState.model_validate_json(
                Path(args.state).read_text(encoding="utf-8")
            )
            if args.state
            else DefenseSchedulerState()
        )
        context = (
            DefenseSchedulingContext.model_validate_json(
                Path(args.context).read_text(encoding="utf-8")
            )
            if args.context
            else None
        )
        schedule = build_defense_resource_schedule(
            plan,
            policy=policy,
            state=state,
            context=context,
        )
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        schedule_path = output / "schedule.json"
        state_path = output / "next-state.json"
        schedule_path.write_text(
            json.dumps(schedule.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(schedule.next_state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "schedule_id": schedule.schedule_id,
                    "schedule_sha256": schedule.schedule_sha256,
                    "scheduled": len(schedule.scheduled),
                    "deferred": len(schedule.deferred),
                    "no_action_components": len(schedule.no_action_component_ids),
                    "scheduling_context": schedule.scheduling_context.context_id,
                    "schedule": str(schedule_path),
                    "next_state": str(state_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-schedule: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
