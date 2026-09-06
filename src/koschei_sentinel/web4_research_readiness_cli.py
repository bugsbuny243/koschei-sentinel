from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.web4_research_readiness import (
    evaluate_web4_research_readiness,
    write_web4_research_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-research-readiness",
        description="Validate the fail-closed Sentinel Web4 research and provenance contract.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/corpus/web4-v1.sources.proposed.jsonl"),
    )
    parser.add_argument(
        "--protocol-tracking",
        type=Path,
        default=Path("configs/corpus/web4-v1.protocol-tracking.json"),
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=Path("configs/curriculum/web4-security.v1.json"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evals/web4-security-benchmark.v1.json"),
    )
    parser.add_argument(
        "--event-schema",
        type=Path,
        default=Path("schemas/web4-security-event-v1.schema.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_web4_research_readiness(
            source_registry_path=args.sources,
            protocol_tracking_path=args.protocol_tracking,
            curriculum_path=args.curriculum,
            benchmark_path=args.benchmark,
            event_schema_path=args.event_schema,
        )
        if args.output is not None:
            write_web4_research_readiness_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"Web4 research readiness rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.ready_for_research else 3


if __name__ == "__main__":
    raise SystemExit(main())
