from __future__ import annotations

import argparse
import json
from pathlib import Path

from koschei_sentinel.cyber_perception import PerceptionBatch
from koschei_sentinel.perception_graph_binding import compile_bound_perception_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile normalized defensive telemetry into a Sentinel Cyber State Graph"
    )
    parser.add_argument("--batch", required=True, help="Perception batch JSON")
    parser.add_argument("--graph-id", required=True, help="Incident Cyber State Graph identifier")
    parser.add_argument("--output", required=True, help="Cyber State Graph JSON")
    parser.add_argument(
        "--receipt-output",
        help="Optional perception graph receipt path; defaults next to --output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        batch = PerceptionBatch.model_validate_json(
            Path(args.batch).read_text(encoding="utf-8")
        )
        bound = compile_bound_perception_graph(batch, graph_id=args.graph_id)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(bound.graph.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        destination.write_text(payload, encoding="utf-8")

        receipt_destination = (
            Path(args.receipt_output)
            if args.receipt_output
            else destination.with_name(f"{destination.stem}.receipt.json")
        )
        receipt_destination.parent.mkdir(parents=True, exist_ok=True)
        receipt_destination.write_text(
            json.dumps(bound.receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(payload, end="")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"sentinel-cyber-perceive: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
