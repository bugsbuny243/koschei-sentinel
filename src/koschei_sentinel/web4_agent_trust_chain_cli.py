from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.web4_agent_trust_chain import (
    evaluate_web4_agent_trust_chain,
    write_web4_agent_trust_chain_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-agent-trust-chain",
        description=(
            "Validate the research-only Web4 Agent Trust Chain source, revision, "
            "benchmark-parent, and authority contract."
        ),
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
        "--benchmark",
        type=Path,
        default=Path("evals/web4-security-benchmark.v1.json"),
    )
    parser.add_argument(
        "--trust-chain-map",
        type=Path,
        default=Path("evals/web4-agent-trust-chain.v1.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_web4_agent_trust_chain(
            source_registry_path=args.sources,
            protocol_tracking_path=args.protocol_tracking,
            benchmark_path=args.benchmark,
            trust_chain_map_path=args.trust_chain_map,
        )
        if args.output is not None:
            write_web4_agent_trust_chain_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"Web4 Agent Trust Chain rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.ready_for_research else 3


if __name__ == "__main__":
    raise SystemExit(main())
