from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.web4_benchmark_intake import (
    build_web4_benchmark_intake,
    write_web4_benchmark_intake_packet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-benchmark-intake",
        description="Build a fail-closed, answer-key-isolated Web4 benchmark intake packet.",
    )
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/corpus/web4-v1.sources.proposed.jsonl"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evals/web4-security-benchmark.v1.json"),
    )
    parser.add_argument(
        "--intake-policy",
        type=Path,
        default=Path("evals/web4-benchmark-intake-policy.v1.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        packet = build_web4_benchmark_intake(
            proposal_path=args.proposal,
            answer_key_path=args.answer_key,
            source_registry_path=args.sources,
            benchmark_policy_path=args.benchmark,
            intake_policy_path=args.intake_policy,
        )
        write_web4_benchmark_intake_packet(packet, args.output)
    except (OSError, ValueError) as exc:
        print(f"Web4 benchmark intake rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(packet.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
