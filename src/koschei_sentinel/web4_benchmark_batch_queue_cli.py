from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.web4_benchmark_batch_queue import build_web4_benchmark_batch_queue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-web4-benchmark-batch-intake",
        description=(
            "Build a fail-closed Web4 benchmark review queue from many proposal/answer-key "
            "pairs while preserving deterministic pre-review splits and answer-key isolation."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
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
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_web4_benchmark_batch_queue(
            manifest_path=args.manifest,
            source_registry_path=args.sources,
            benchmark_policy_path=args.benchmark,
            intake_policy_path=args.intake_policy,
            snapshot_root=args.snapshot_root,
            output_dir=args.output_dir,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"Web4 benchmark batch intake rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
