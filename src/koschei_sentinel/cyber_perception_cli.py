from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_perception import PerceptionBatch, compile_perception_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile normalized defensive telemetry into a Sentinel Cyber State Graph"
    )
    parser.add_argument("--batch", required=True, help="Perception batch JSON")
    parser.add_argument("--graph-id", required=True, help="Incident Cyber State Graph identifier")
    parser.add_argument("--output", required=True, help="Cyber State Graph JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        batch = PerceptionBatch.model_validate_json(
            Path(args.batch).read_text(encoding="utf-8")
        )
        graph = compile_perception_batch(batch, graph_id=args.graph_id)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(graph.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-perceive: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
