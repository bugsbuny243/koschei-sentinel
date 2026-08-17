from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.active_defense_planner import build_active_defense_plan
from koschei_sentinel.cyber_state_graph import CyberStateGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a Sentinel Cyber State Graph and emit a fail-closed active-defense plan"
        )
    )
    parser.add_argument("--graph", required=True, help="Cyber State Graph JSON file")
    parser.add_argument(
        "--critical",
        action="append",
        default=[],
        help="Protected critical entity id; may be repeated",
    )
    parser.add_argument("--output", help="Optional output JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        graph = CyberStateGraph.model_validate_json(Path(args.graph).read_text(encoding="utf-8"))
        plan = build_active_defense_plan(graph, critical_entity_ids=args.critical)
        payload = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-active-defense-plan: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
