from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.assured_multi_incident_defense import AssuredMultiIncidentDefensePlan
from koschei_sentinel.defense_resource_scheduler import DefenseResourceSchedule
from koschei_sentinel.defense_wave_execution import (
    DefenseWaveExecution,
    authorize_wave_component,
    record_wave_component_execution,
    start_defense_wave_execution,
    verify_wave_component_outcome,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coordinate auditable parallel Sentinel defense-wave execution"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start a wave from an exact defense schedule and plan")
    start.add_argument("--schedule", required=True)
    start.add_argument("--plan", required=True)
    start.add_argument("--output", required=True)

    authorize = sub.add_parser("authorize", help="Authorize one scheduled component step")
    authorize.add_argument("--wave", required=True)
    authorize.add_argument("--component", required=True)
    authorize.add_argument("--evidence", action="append", required=True)
    authorize.add_argument("--output", required=True)

    record = sub.add_parser("record", help="Record connector execution receipts for one component")
    record.add_argument("--wave", required=True)
    record.add_argument("--component", required=True)
    record.add_argument("--receipt", action="append", required=True)
    record.add_argument("--output", required=True)

    verify = sub.add_parser("verify", help="Verify one component outcome with evidence")
    verify.add_argument("--wave", required=True)
    verify.add_argument("--component", required=True)
    outcome = verify.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--succeeded", action="store_true")
    outcome.add_argument("--failed", action="store_true")
    verify.add_argument("--evidence", action="append", required=True)
    verify.add_argument("--output", required=True)
    return parser


def _read_wave(path: str) -> DefenseWaveExecution:
    return DefenseWaveExecution.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _write_wave(path: str, wave: DefenseWaveExecution) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(wave.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "start":
            schedule = DefenseResourceSchedule.model_validate_json(
                Path(args.schedule).read_text(encoding="utf-8")
            )
            plan = AssuredMultiIncidentDefensePlan.model_validate_json(
                Path(args.plan).read_text(encoding="utf-8")
            )
            wave = start_defense_wave_execution(schedule, plan)
        elif args.command == "authorize":
            wave = authorize_wave_component(
                _read_wave(args.wave),
                component_id=args.component,
                precondition_evidence_ids=list(dict.fromkeys(args.evidence)),
            )
        elif args.command == "record":
            wave = record_wave_component_execution(
                _read_wave(args.wave),
                component_id=args.component,
                execution_receipt_ids=list(dict.fromkeys(args.receipt)),
            )
        else:
            wave = verify_wave_component_outcome(
                _read_wave(args.wave),
                component_id=args.component,
                succeeded=bool(args.succeeded),
                outcome_evidence_ids=list(dict.fromkeys(args.evidence)),
            )
        _write_wave(args.output, wave)
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-defense-wave: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
