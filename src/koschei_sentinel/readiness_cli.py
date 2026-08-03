from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koschei_sentinel.readiness import (
    ReadinessPolicy,
    evaluate_release_readiness,
    load_readiness_policy,
    write_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-dataset-readiness",
        description="Evaluate whether a Sentinel dataset release is ready for real training.",
    )
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = load_readiness_policy(args.policy) if args.policy else ReadinessPolicy()
        report = evaluate_release_readiness(args.release, policy=policy)
        if args.output is not None:
            write_readiness_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"readiness rejected: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
