from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.attack_world_lines import (
    AttackWorldLinePolicy,
    build_attack_world_line_timeline,
)
from koschei_sentinel.cyber_state_graph import CyberStateGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track independent Sentinel attack components across graph snapshots"
    )
    parser.add_argument(
        "--graph",
        action="append",
        required=True,
        help="Cyber State Graph snapshot JSON; repeat in chronological order",
    )
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--policy", help="Optional attack world-line policy JSON")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshots = [
            CyberStateGraph.model_validate_json(Path(path).read_text(encoding="utf-8"))
            for path in args.graph
        ]
        policy = (
            AttackWorldLinePolicy.model_validate_json(
                Path(args.policy).read_text(encoding="utf-8")
            )
            if args.policy
            else AttackWorldLinePolicy()
        )
        timeline = build_attack_world_line_timeline(
            snapshots,
            stream_id=args.stream_id,
            policy=policy,
        )
        payload = json.dumps(
            timeline.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ) + "\n"
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-attack-world-lines: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
